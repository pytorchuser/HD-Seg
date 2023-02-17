# Copyright (c) OpenMMLab. All rights reserved.
import time
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.utils import get_root_logger
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM


@HEADS.register_module()
class UPerHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(UPerHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.t = time.time()
        self.logger = get_root_logger()
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        t1 = time.time()
        # self.logger.info(f'psp module初始化耗时：{t1-self.t}, 累计总时长：{t1-self.t}')
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        t2 = time.time()
        # self.logger.info(f'fpn module初始化耗时：{t2 - t1}, 累计总时长：{t2 - self.t}')

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        t = time.time()
        # -1, x 就是 inputs 的最后的特征
        x = inputs[-1]
        # 原图 x 及其 进行psp的特征都放入 列表中进行保存
        psp_outs = [x]  # 原图 x
        psp_outs.extend(self.psp_modules(x))  # 返回的4个pmp block的输出
        # 把他们拼在一起后: psp_outs: [2, 2816, 16, 16]
        psp_outs = torch.cat(psp_outs, dim=1)
        # 拼完后的结构再进行一次 3*3的卷积，把输出的channel从2816给降维到512，返回结果到UPerHead的 forward中
        output = self.bottleneck(psp_outs)
        # self.logger.info(f'psp_forward执行一次耗时：{time.time() - t}, 累计总时长：{time.time() - self.t}')
        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # inputs 是一个list ， 即前面4个stage得到的结果, 每个特征channels不同， 即特征维度不同
        inputs = self._transform_inputs(inputs)

        # build laterals 对inputs进行卷积，让特征图有一个一致的维度
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        t = time.time()
        # 深层的特征单独拿出来进行psp forward，在psp forward中，取了input的最后一层
        laterals.append(self.psp_forward(inputs))
        t1 = time.time()
        # self.logger.info(f'循环laterals.append耗时：{t1 - t}, 累计总时长：{t1 - self.t}')
        # build top-down path
        used_backbone_levels = len(laterals)

        # 把深层特征进行psp forward后(16, 16)再进行上采样，与前面stage输出的浅层特征进行残差连接（加和）（32，32）
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:] #浅层stage输出的尺寸:[32,32]
            # laterals[i - 1]即前面stage输出的浅层特征， resize对特征进行上采样
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)
        t2 = time.time()
        # self.logger.info(f'循环laterals.insert耗时：{t2 - t1}, 累计总时长：{t2 - self.t}')
        # 这里的遍历是[32] [64] [128]的3个残差连接后的特征图再各自走了一个卷积
        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        t3 = time.time()
        # self.logger.info(f'fpn_outs耗时：{t3 - t2}, 累计总时长：{t3 - self.t}')
        # 把psp那个[16,16]的特征图也加进来成为4个特征图
        # append psp feature
        fpn_outs.append(laterals[-1])

        # 把4个特征图的尺寸通过上采样进行统一，为 128
        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        # 把各个stage的特征进行fuse， fpn_outs [2,2816,128,128]
        fpn_outs = torch.cat(fpn_outs, dim=1)
        # 融合后再进行一次卷积,把2816降维成512,output:[2,512,128,128]
        feats = self.fpn_bottleneck(fpn_outs)
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
