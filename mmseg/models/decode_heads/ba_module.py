import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from mmcv.cnn import build_conv_layer, build_upsample_layer, ConvModule, build_activation_layer, build_norm_layer
from mmengine.model import BaseModule
# from ..utils.attention import AttLayer, SKLayer


class UpBlock(BaseModule):
    def __init__(self,
                 in_channels,
                 scale_factor=2,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super().__init__()
        if conv_cfg is None:
            conv_cfg = dict(type='Conv2d')
        if norm_cfg is None:
            norm_cfg = dict(type='BN')
        if act_cfg is None:
            act_cfg = dict(type='ReLU')

        self.up_sample = build_upsample_layer(
            cfg=dict(scale_factor=scale_factor,
                     type='bilinear',
                     align_corners=False))
        self.up_conv = ConvModule(
            in_channels,
            in_channels // scale_factor,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        """
            Input: (B, C, H, W)
            Output: (B, C_out, H_out, W_out)
        """
        x = self.up_sample(x)  # (B, C, H_out, W_out)
        x = self.up_conv(x)  # (B, C_out, H_out, W_out)
        return x


class STN(BaseModule):
    def __init__(self,
                 in_channels,
                 conv_cfg=None,
                 act_cfg=None):
        super().__init__()
        if conv_cfg is None:
            conv_cfg = dict(type='Conv2d')
        if act_cfg is None:
            act_cfg = dict(type='ReLU')
        # loc 定义了两层卷积网络
        self.loc = nn.Sequential(
            # 卷积输出shape为(B,8,H_out,W_out)
            build_conv_layer(
                cfg=conv_cfg,
                in_channels=in_channels,
                out_channels=8,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            # 最大池化输出shape为(B,8,H_out/2,W_out/2)
            nn.MaxPool2d(kernel_size=2, stride=2),
            build_activation_layer(act_cfg),
            # 卷积输出shape为(B,10,H_o,W_o)
            build_conv_layer(
                cfg=conv_cfg,
                in_channels=8,
                out_channels=10,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            # 最大池化输出shape为(B,10,H_o/2,W_o/2)
            nn.MaxPool2d(kernel_size=2, stride=2),
            build_activation_layer(act_cfg)
        )
        # 采用两层全连接层，回归出仿射变换所需的参数θ（6，）
        self.fc_loc = nn.Sequential(
            # nn.Linear(self.x_size, 32),
            build_activation_layer(act_cfg),
            nn.Linear(32, 2 * 3)
        )

        # Initialize the weights/bias with identity transformation
        # self.fc_loc[1].weight.data.zero_()
        # self.fc_loc[1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        """
            Input: (B, C, H, W)
        """
        batch_size = x.size(0)
        # 提取输入图像中的特征
        xs = self.loc(x)
        xs = xs.view(batch_size, 10 * xs.size(2) * xs.size(3))  # （B, 10 * H_xs * W_xs)
        # 回归theta参数
        # 以下对应nn.Linear(10 * 3 * 3, 32)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        weight = Parameter(torch.empty((32, xs.size(1)))).to(device)
        bias = Parameter(torch.empty(32)).to(device)
        init.kaiming_uniform_(weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(bias, -bound, bound)
        theta = F.linear(xs, weight, bias)

        theta = self.fc_loc(theta)
        theta = theta.view(batch_size, 2, 3)  # (B, C, 3)
        # Grid Generator
        grid = F.affine_grid(theta, x.size())  # (B, H_out, W_out, C)
        # Bilinear Sampler
        warp = F.grid_sample(x, grid)  # (B, C, H_out, W_out)
        return warp


class BA(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 act_cfg=None,
                 norm_cfg=None,
                 conv_cfg=None):
        super().__init__()
        if norm_cfg is None:
            norm_cfg = dict(type='BN', requires_grad=True)
        if act_cfg is None:
            act_cfg = dict(type='ReLU')
        if conv_cfg is None:
            conv_cfg = dict(type='Conv2d')
        self.out_channels = out_channels
        self.up_sample = UpBlock(in_channels * 2)
        self.c_2_conv = build_conv_layer(
            cfg=conv_cfg,
            in_channels=in_channels * 2,
            out_channels=in_channels,
            kernel_size=1)
        self.stn = STN(in_channels)
        # self.re_conv = build_conv_layer(
        #     cfg=conv_cfg,
        #     in_channels=2,
        #     out_channels=in_channels,
        #     kernel_size=1)

        # 初始化ea层中的attLayer
        # self.att_layer = AttLayer(in_channels=in_channels)

        self.re_2_conv = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                in_channels * 2,
                self.out_channels,
                kernel_size=1,
                bias=False),
            build_norm_layer(norm_cfg, in_channels)[1],
            build_activation_layer(act_cfg)
        )

    def forward(self, x, x_low):
        """
            Input: (B, C, H, W), (B, C_low, H_low, W_low)
            Output: (B, C, H, W)
        """
        # x_low用双线性上采样至跟x尺寸一致
        x_low = self.up_sample(x_low)
        # concat
        out = torch.cat([x, x_low], dim=1)
        # 用3*3卷积将管道改为2
        out = self.c_2_conv(out)
        # 过STN
        out = self.stn(out)
        # 将管道数改为原图大小
        # out = self.re_conv(out)
        # 用原图-STN输出得到最后的输出
        # x = self.stn(x)
        out = x - out

        out = torch.cat([x, out], dim=1)
        out = self.re_2_conv(out)

        # # att流程
        # out = self.att_layer(out)

        return out
