import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_activation_layer, build_norm_layer, ConvModule
from mmengine.model import BaseModule

from mmseg.models.utils import StripPooling


class Flatten(BaseModule):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelAtt(BaseModule):
    def __init__(self,
                 gate_channels,
                 reduction_ratio=2,
                 pool_types=None,
                 act_cfg=None):
        super(ChannelAtt, self).__init__()
        if pool_types is None:
            pool_types = ['avg', 'max', 'soft']
        if act_cfg is None:
            act_cfg = dict(type='ReLU')
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            build_activation_layer(act_cfg),
        )
        self.pool_types = pool_types
        self.incr = nn.Linear(gate_channels // reduction_ratio, gate_channels)

    def forward(self, x):
        avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        avg_pool_mlp = self.mlp(avg_pool)
        max_pool_mlp = self.mlp(max_pool)

        # release
        # soft_pool = SoftPool2d(kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        # soft_pool=soft_pool(x)
        # soft_pool_mlp = self.mlp(soft_pool)
        sum_pool = avg_pool_mlp + max_pool_mlp

        channel_att_sum = self.incr(sum_pool)
        att = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(avg_pool)
        return att


class ESChannelAtt(BaseModule):
    """
        input (B, C, H, W)
        output (B, C, H, W)
    """
    def __init__(self,
                 in_channels,
                 r=2,
                 conv_cfg=None, act_cfg=None):
        super().__init__()
        if act_cfg is None:
            act_cfg = dict(type='ReLU')
        if conv_cfg is None:
            conv_cfg = dict(type='Conv2d')

        self.conv = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                in_channels,
                in_channels // r,
                1),
            build_activation_layer(act_cfg),
            build_conv_layer(
                conv_cfg,
                in_channels // r,
                in_channels,
                1)
        )

    def forward(self, x):
        avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        out1 = self.conv(avg_pool)
        out1 = torch.sigmoid(out1)
        out = out1 * x
        return out


class BranchAtt(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ratio=4,
                 branch_num=2,
                 lower_bound=32,
                 conv_type='Conv2d',
                 act_cfg=None):
        super().__init__()
        if act_cfg is None:
            act_cfg = dict(type='ReLU', inplace=True)
        self.out_channels = out_channels
        self.branch_num = branch_num
        # 计算出降维参数d
        d = max(in_channels // ratio, lower_bound)
        # mlp操作，降维
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(in_channels, d),
            build_activation_layer(act_cfg),
        )
        # 升维
        self.asc_de = build_conv_layer(
                dict(type=conv_type),
                in_channels=d,
                out_channels=out_channels * branch_num,
                kernel_size=1,
                stride=1,
                bias=False)
        self.soft_max = nn.Softmax(dim=1)  # 令两个FCs每个维度上的和为1， 即 a+b=1

    def forward(self, x):
        # batch_size = x.shape(0)
        batch_size = x.shape[0]
        # 先降维再升维，reshape成按branch_num可拆分的形状
        mlp_out = self.mlp(x)
        x = mlp_out.unsqueeze(2).unsqueeze(3)
        x = self.asc_de(x)  # (B, C*2, H, W)
        x = x.reshape(batch_size, self.branch_num, self.out_channels, -1)  # (B, 2, C ,H*W)
        # 走softmax，chunk成需要的块后再reshape可处理的形状
        x = self.soft_max(x)
        a_b = list(x.chunk(self.branch_num, dim=1))
        a_b = list(map(lambda a: a.reshape(batch_size, self.out_channels, 1, 1), a_b))
        return a_b


class AttLayer(BaseModule):
    """
    input (B, C, H, W)
    """
    def __init__(self,
                 in_channels,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.channel_att = ChannelAtt(gate_channels=in_channels, reduction_ratio=2)
        self.strip_pool = StripPooling(in_channels, (20, 12))

    def forward(self, x):
        # 1。channel att
        c_att = self.channel_att(x)
        att = c_att * x
        # 2。空间 att
        att = self.strip_pool(att)
        x = att + x
        return x


class SKLayer(BaseModule):
    """
        input (B, C, H, W)
    """
    def __init__(self,
                 in_channels):
        super().__init__()
        self.branch_att = BranchAtt(in_channels, in_channels)

    def forward(self, swin_out, res_out):
        # swin+res
        x = swin_out + res_out
        # 平均池化
        avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        # 分支att（mlp+softmax）
        a_b = self.branch_att(avg_pool)
        # 两个分别乘 swin和res
        swin = swin_out * a_b[0]
        res = res_out * a_b[1]
        # 再相加
        out = swin + res
        return out


class Residual(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels, conv_cfg=None, act_cfg=None, norm_cfg=None):
        super().__init__()

        if norm_cfg is None:
            norm_cfg = dict(type='BN', requires_grad=True)
        if act_cfg is None:
            act_cfg = dict(type='ReLU')
        if conv_cfg is None:
            conv_cfg = dict(type='Conv2d')

        self.skip_layer = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            1)

        self.conv = nn.Sequential(
            build_norm_layer(norm_cfg, in_channels)[1],
            build_activation_layer(act_cfg),
            build_conv_layer(
                conv_cfg,
                in_channels,
                out_channels // 2,
                1),
            build_norm_layer(norm_cfg, out_channels // 2)[1],
            build_activation_layer(act_cfg),
            build_conv_layer(
                conv_cfg,
                out_channels // 2,
                out_channels // 2,
                3),
            build_norm_layer(norm_cfg, out_channels // 2)[1],
            build_activation_layer(act_cfg),
            build_conv_layer(
                conv_cfg,
                out_channels // 2,
                out_channels,
                1),
        )

    def forward(self, x):
        out_1 = self.skip_layer(x)
        out_2 = self.conv(x)
        out = out_1 + out_2
        return out


class BiFLayer(BaseModule):
    """
        input (B, C, H, W)
        https://github.com/Rayicer/TransFuse/blob/main/imgs/model.png
    """
    def __init__(self,
                 in_channels,
                 c_att_r=2,
                 conv_cfg=None,
                 act_cfg=None,
                 norm_cfg=None):
        super().__init__()
        if norm_cfg is None:
            norm_cfg = dict(type='BN', requires_grad=True)
        if act_cfg is None:
            act_cfg = dict(type='ReLU')
        if conv_cfg is None:
            conv_cfg = dict(type='Conv2d')
        self.w_conv = ConvModule(
            in_channels,
            in_channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.strip_pool = StripPooling(in_channels)
        self.channel_att = ESChannelAtt(in_channels, c_att_r)
        self.residual = Residual(in_channels * 3, in_channels)

    def forward(self, swin_out, res_out):
        # bilinear pooling
        w_product = swin_out * res_out
        w_product = self.w_conv(w_product)

        # spatial attention for resnet
        g_out = self.strip_pool(res_out)

        # channel attention for swin
        t_out = self.channel_att(swin_out)

        # residual
        out = self.residual(torch.cat([g_out, t_out, w_product], dim=1))
        return out
