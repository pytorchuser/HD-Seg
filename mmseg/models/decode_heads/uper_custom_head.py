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
class UPerCustomHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(UPerCustomHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.t = time.time()
        self.logger = get_root_logger()
        # PSP Module
        # 读取配置文件的参数，初始化PPM或其他(UFE)处理模型列表
        # 初始化对应的bottleneck卷积模型列表
        self.fe_modules, self.bottleneck_modules = self.init_fe_bottleneck_modules(pool_scales)
        t1 = time.time()
        # self.logger.info(f'psp module初始化耗时：{t1-self.t}, 累计总时长：{t1-self.t}')
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        # 获取不需要特殊处理的层级idx
        if self.msc_module_cfg is None:
            self.norm_idx = self.in_index
        else:
            invalid_idx = [msc.layer_idx for msc in self.msc_module_cfg]
            self.norm_idx = list(set(self.in_index).difference(set(invalid_idx)))
        # 如果所有特征层都需要进行模型处理则不需要生成如下卷积模型。
        # 只有不需要进行特殊处理的特征层才会生成对应的l_conv和fpn_conv
        for i in self.norm_idx:  # 跳过特殊处理的层
            l_conv = ConvModule(
                self.in_channels[i],
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            # ？搞不明白，为什么未处理层不能与同一个fpn_conv卷积操作，而是要生成多个相同的fpn_conv分别进行卷积
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

        # fpn_bottleneck对拼接后的特征图进行卷积操作，以得到目标out channel数
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

    # inputs: transformer 每个stage输出的特征图列表
    # idx： 特殊处理的层级layer_idx inputs[idx]
    # module_idx： idx对应卷积module的下标 fe_modules[module_idx]
    # 用idx和module_idx确保卷积层in_channel一一对应
    def fe_forward(self, inputs, idx, module_idx):
        """Forward function of PSP or other module."""
        t = time.time()
        # -1, x 就是 inputs 的最后的特征
        x = inputs[idx]
        # 原图 x 及其 进行psp的特征都放入 列表中进行保存
        psp_outs = [x]  # 原图 x
        # 添加根据idx来处理的过程，注意psp_modules[]要取对应下标的
        psp_outs.extend(self.fe_modules[module_idx](x))  # 返回的4个pmp block的输出
        # 把他们拼在一起后: psp_outs: [2, 2816, 16, 16]
        psp_outs = torch.cat(psp_outs, dim=1)
        # 拼完后的结构再进行一次 3*3的卷积，把输出的channel从2816给降维到512，返回结果到UPerHead的 forward中
        output = self.bottleneck_modules[module_idx](psp_outs)
        # self.logger.info(f'fe_forward执行一次耗时：{time.time() - t}, 累计总时长：{time.time() - self.t}')
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

        # build laterals 对inputs进行卷积，让norm特征图有一个一致的维度
        laterals = [
            self.lateral_convs[i](inputs[idx])
            for i, idx in enumerate(self.norm_idx)
        ]

        t = time.time()
        # 需要处理的特征图拿出来进行psp(fe) forward
        # 根据配置文件-特殊处理层数来决定，对特征图哪一层进行什么类型的模型处理。处理后再使用下标插值，存入laterals对应位置，保证层级顺序
        if self.msc_module_cfg is not None:
            for i in range(len(self.msc_module_cfg)):
                msc_layer_idx = self.msc_module_cfg[i].layer_idx
                laterals.insert(msc_layer_idx, self.fe_forward(inputs, msc_layer_idx, i))
        t1 = time.time()
        # self.logger.info(f'循环laterals.insert耗时：{t1 - t}, 累计总时长：{t1 - self.t}')

        # build top-down path
        used_backbone_levels = len(laterals)

        # ？ FPN功能
        # 把深层特征进行psp forward后(16, 16)再进行上采样，与前面stage输出的浅层特征进行残差连接（加和）（32，32）
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]  # 浅层stage输出的尺寸:[32,32]
            # laterals[i - 1]即前面stage输出的浅层特征， resize对特征进行上采样
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)
        t2 = time.time()
        # self.logger.info(f'循环laterals.insert耗时：{t2 - t1}, 累计总时长：{t2 - self.t}')
        # build outputs
        # 对不需要特殊模型处理的层数残差连接后的特征图再各自走了一个卷积
        # 对不需要特殊处理的特征图进行3*3卷积并放入fpn_outs中
        # todo 查找3*3卷积的意义并确认该操作是否属于FPN方法
        fpn_outs = [
            self.fpn_convs[i](laterals[idx])
            for i, idx in enumerate(self.norm_idx)
        ]
        t3 = time.time()
        # self.logger.info(f'fpn_outs耗时：{t3 - t2}, 累计总时长：{t3 - self.t}')
        # 把特殊处理过的特征图也加进来成为4个特征图
        # append psp feature
        # 根据配置文件-特殊处理层数，将特殊处理的特征图直接存入fpn_outs
        if self.msc_module_cfg is not None:
            for msc_module in self.msc_module_cfg:
                fpn_outs.insert(msc_module.layer_idx, laterals[msc_module.layer_idx])

        # 把4个特征图的尺寸通过上采样进行统一，为 128
        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        # 把各个stage的特征进行fuse， fpn_outs [2,2048,128,128]
        fpn_outs = torch.cat(fpn_outs, dim=1)
        # 融合后再进行一次卷积,把2048降维成512,output:[2,512,128,128]
        feats = self.fpn_bottleneck(fpn_outs)
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output

    # 读取配置文件的参数，来决定是使用PPM或其他(UFE)模型对指定图像层进行处理
    # 返回一个feature extraction moduleList，按顺序存放生成的处理模型
    # 返回一个bottleneck moduleList，按顺序存放生成的
    def init_fe_bottleneck_modules(self, pool_scales):
        if self.msc_module_cfg is None:
            return None, None
        fe_modules = nn.ModuleList()
        bottleneck_modules = nn.ModuleList()
        # msc_module_cfg=[dict(type='PPM', layer_idx=3), dict(type='MSC', layer_idx=2]
        for i in range(len(self.msc_module_cfg)):
            layer_idx = self.msc_module_cfg[i].layer_idx
            if self.msc_module_cfg[i].type == 'PPM':
                # 根据下标取对应层级的模型（确保self.in_channels[-1]能一一对应）
                fe_module = PPM(
                    pool_scales,
                    self.in_channels[layer_idx],
                    self.channels,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    align_corners=self.align_corners)
            # TODO 新增其他类型的优化模块时，在此处添加elif，初始化不同类型的功能模块存入列表，后续fe_forward()要用

                fe_modules.append(fe_module)

            # 生成对应的bottleneck，in_channels[i]需对应
            bottleneck = ConvModule(
                self.in_channels[layer_idx] + len(pool_scales) * self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            bottleneck_modules.append(bottleneck)

        return fe_modules, bottleneck_modules
