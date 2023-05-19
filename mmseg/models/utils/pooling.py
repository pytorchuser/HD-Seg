import torch
from torch import nn
from torch.nn import functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer, build_activation_layer
from mmengine.model import BaseModule


class StripPooling(BaseModule):
    def __init__(self,
                 in_channels,
                 pool_size,
                 norm_cfg=None,
                 act_cfg=None,
                 conv_cfg=None):
        super().__init__(norm_cfg)
        if norm_cfg is None:
            norm_cfg = dict(type='BN', requires_grad=True)
        if act_cfg is None:
            act_cfg = dict(type='ReLU')
        if conv_cfg is None:
            conv_cfg = dict(type='Conv2d')
        # 双线性插值
        self.up_cfg = dict(mode='bilinear', align_corners=True)
        # 改变管道数的1*1卷积
        inter_channels = in_channels // 4
        self.c_conv = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                in_channels,
                inter_channels,
                kernel_size=1,
                bias=False),
            build_norm_layer(norm_cfg, inter_channels)[1],
            build_activation_layer(act_cfg)
        )
        # 一个普通的3*3卷积
        self.conv3_3 = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                inter_channels,
                inter_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            build_norm_layer(norm_cfg, inter_channels)[1],
        )
        # 使用pool_size[0]池化操作并进行卷积
        self.pool_0_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size[0]),
            build_conv_layer(
                conv_cfg,
                inter_channels,
                inter_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            build_norm_layer(norm_cfg, inter_channels)[1],
        )
        # 使用pool_size[1]池化操作并进行卷积
        self.pool_1_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size[1]),
            build_conv_layer(
                conv_cfg,
                inter_channels,
                inter_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            build_norm_layer(norm_cfg, inter_channels)[1],
        )
        self.pool_h_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            build_conv_layer(
                conv_cfg,
                inter_channels,
                inter_channels,
                kernel_size=(3, 1),
                stride=1,
                padding=(1, 0),
                bias=False),
            build_norm_layer(norm_cfg, inter_channels)[1],
        )
        self.pool_w_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            build_conv_layer(
                conv_cfg,
                inter_channels,
                inter_channels,
                kernel_size=(1, 3),
                stride=1,
                padding=(0, 1),
                bias=False),
            build_norm_layer(norm_cfg, inter_channels)[1],
        )
        # 处理相加结果的3*3卷积
        self.conv = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                inter_channels,
                inter_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            build_norm_layer(norm_cfg, inter_channels)[1],
            build_activation_layer(act_cfg)
        )
        # 还原管道数的卷积
        self.re_conv = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                inter_channels * 2,
                in_channels,
                kernel_size=1,
                bias=False),
            build_norm_layer(norm_cfg, inter_channels)[1],
        )

    def forward(self, x):
        _, _, h, w = x.size()
        # 改变管道大小的卷积
        x1 = x2 = self.c_conv(x)
        x1_2 = self.conv3_3(x1)
        # x1 池化-3*3卷积-扩充（两遍走不同尺寸的池化）
        x1_0 = F.interpolate(self.pool_0_conv(x1), (h, w), **self.up_cfg)
        x1_1 = F.interpolate(self.pool_1_conv(x1), (h, w), **self.up_cfg)
        # 将x1经过以上不同流程的值相加
        x1 = self.conv(F.relu_(x1_0 + x1_1 + x1_2))
        # x2 池化-卷积-扩充
        x2_h = F.interpolate(self.pool_h_conv(x2), (h, w), **self.up_cfg)
        x2_w = F.interpolate(self.pool_w_conv(x2), (h, w), **self.up_cfg)
        # 将x2经过以上不同流程的值相加
        x2 = self.conv(F.relu_(x2_h + x2_w))
        # 将x1 x2拼在一起并还原管道数
        out = self.re_conv(torch.cat([x1, x2], dim=1))
        # out与input原值直接相加
        out = F.relu_(x + out)
        return out
