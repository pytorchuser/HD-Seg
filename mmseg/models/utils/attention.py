import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_activation_layer
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
        # 降维
        self.re_de = nn.Sequential(
            build_conv_layer(
                dict(type=conv_type),
                in_channels=out_channels,
                out_channels=d,
                kernel_size=1,
                bias=False),
            nn.BatchNorm2d(d),
            build_activation_layer(act_cfg)
        )
        # 升维
        self.asc_de = build_conv_layer(
                dict(type=conv_type),
                in_channels=d,
                out_channels=out_channels * branch_num,
                kernel_size=1,
                stride=1,
                bias=False)
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, x):
        # batch_size = x.shape(0)
        batch_size = x.shape[0]
        # 先降维再升维，reshape成按branch_num可拆分的形状
        x = self.re_de(x)
        x = self.asc_de(x)
        x = x.reshape(batch_size, self.branch_num, self.out_channels, -1)
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

