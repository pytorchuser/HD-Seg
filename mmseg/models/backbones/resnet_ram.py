# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer, build_activation_layer
from mmcv.ops import DeformConv2dPack
from mmengine.model import BaseModule, ModuleList
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from SoftPool import SoftPool2d

from ..utils import ResLayer, StripPooling
from ..backbones.resnet import BasicBlock, Bottleneck


def make_res_layer(**kwargs):
    """Pack all blocks in a stage into a ``ResLayer``."""
    return ResLayer(**kwargs)


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
        soft_pool = SoftPool2d(kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        soft_pool = self.mlp(soft_pool(x))
        sum_pool = avg_pool_mlp + max_pool_mlp + soft_pool

        # debug 用
        # test_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        # test_pool = self.mlp(test_pool)
        # sum_pool = avg_pool_mlp + max_pool_mlp + test_pool

        channel_att_sum = self.incr(sum_pool)
        att = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
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
        batch_size = x.shape(0)
        # 先降维再升维，reshape成按branch_num可拆分的形状
        x = self.re_de(x)
        x = self.asc_de(x)
        x = x.reshape(batch_size, self.branch_num, self.out_channels, -1)
        # 走softmax，chunk成需要的块后再reshape可处理的形状
        x = self.soft_max(x)
        a_b = list(x.chunk(self.branch_num, dim=1))
        a_b = list(map(lambda a: a.reshape(batch_size, self.out_channels, 1, 1), a_b))
        return a_b


class RamLayer(BaseModule):
    """
    input (B, C, H, W)
    """
    def __init__(self,
                 in_channel,
                 out_channel,
                 is_swin_ram=False,
                 is_res_ram=False,
                 ram_simple=True,
                 init_cfg=None,
                 act_cfg=None,
                 conv_type='Conv2d'):
        super().__init__(init_cfg)
        if act_cfg is None:
            act_cfg = dict(type='ReLU')
        self.is_swin_ram = is_swin_ram
        self.is_res_ram = is_res_ram
        self.ram_simple = ram_simple
        # RAM初始化
        self.s_conv = nn.Sequential(
            # 1*1卷积
            build_conv_layer(
                dict(type=conv_type),
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=1,
                stride=1),
            nn.BatchNorm2d(out_channel, eps=1e-5, momentum=0.01, affine=True),
            build_activation_layer(act_cfg)
        )
        self.channel_att = ChannelAtt(gate_channels=out_channel, reduction_ratio=2)
        # self.ram_dcn = DeformConv2dPack(out_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.branch_att = BranchAtt(out_channel, out_channel)
        self.strip_pool_res = StripPooling(out_channel, (20, 12))
        self.strip_pool_swin = StripPooling(out_channel, (20, 12))

    def forward(self, x, net_out):
        if self.is_swin_ram:
            # swin是主网络
            # resnet对应stage输出做1*1卷积，改变管道数
            net_out = self.s_conv(net_out)
            if self.ram_simple:
                # TODO 验证'+'的效果
                x = x + net_out
            else:
                x = x + net_out
                # 1。channel att
                # sum分别走三个pool+mlp，三个结果相加后走sigmoid
                x = self.channel_att(x)
                # 2。分支 branch attention
                a_b = self.branch_att(x)
                # 将out与分支注意力结果，按顺序分别相乘
                swin_out = x * a_b[0]
                res_out = net_out * a_b[1]
                # 3。空间
                # 将out分别走stripPool
                swin_out = self.strip_pool_swin(swin_out)
                res_out = self.strip_pool_res(res_out)
                # 再相加
                x = swin_out + res_out
        elif self.is_res_ram:
            # res是主网络
            # swin对应stage输出做1*1卷积，改变管道数
            net_out = self.s_conv(net_out)
            if self.ram_simple:
                # TODO 验证'+'的效果
                x = x + net_out
            else:
                x = self.strip_pool_res(x)
                out = self.strip_pool_swin(net_out)
                x = x + out
                # # DCN 可变形卷积
                # x = self.ram_dcn(x)
                # # 对swin做fuse
                # p = self.s_fuse(net_out)
                # # 将两个结果相乘
                # p = p * x
                # # 三部分相加
                # x = x + p + net_out
        return x


class ResNetRam(BaseModule):
    """ResNet backbone.

    This backbone is the improved implementation of `Deep Residual Learning
    for Image Recognition <https://arxiv.org/abs/1512.03385>`_.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        stem_channels (int): Number of stem channels. Default: 64.
        base_channels (int): Number of base channels of res layer. Default: 64.
        num_stages (int): Resnet stages, normally 4. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: (1, 2, 2, 2).
        dilations (Sequence[int]): Dilation of each stage.
            Default: (1, 1, 1, 1).
        out_indices (Sequence[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default: 'pytorch'.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Default: False.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict | None): Dictionary to construct and config conv layer.
            When conv_cfg is None, cfg will be set to dict(type='Conv2d').
            Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        dcn (dict | None): Dictionary to construct and config DCN conv layer.
            When dcn is not None, conv_cfg must be None. Default: None.
        stage_with_dcn (Sequence[bool]): Whether to set DCN conv for each
            stage. The length of stage_with_dcn is equal to num_stages.
            Default: (False, False, False, False).
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.

            - position (str, required): Position inside block to insert plugin,
            options: 'after_conv1', 'after_conv2', 'after_conv3'.

            - stages (tuple[bool], optional): Stages to apply plugin, length
            should be same as 'num_stages'.
            Default: None.
        multi_grid (Sequence[int]|None): Multi grid dilation rates of last
            stage. Default: None.
        contract_dilation (bool): Whether contract first dilation of each layer
            Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: True.
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.

    Example:
        >>> from mmseg.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 swin_channels=96,
                 in_channels=3,
                 stem_channels=32,
                 base_channels=32,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=None,
                 norm_eval=False,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 multi_grid=None,
                 contract_dilation=False,
                 with_cp=False,
                 zero_init_residual=True,
                 pretrained=None,
                 init_cfg=None,
                 is_res_ram=False,
                 ram_simple=True):
        super().__init__(init_cfg)
        if norm_cfg is None:
            norm_cfg = dict(type='BN', requires_grad=True)
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')

        self.pretrained = pretrained
        self.zero_init_residual = zero_init_residual
        block_init_cfg = None
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
                block = self.arch_settings[depth][0]
                if self.zero_init_residual:
                    if block is BasicBlock:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm2'))
                    elif block is Bottleneck:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm3'))
        else:
            raise TypeError('pretrained must be a str or None')

        self.depth = depth
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert 1 <= num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.multi_grid = multi_grid
        self.contract_dilation = contract_dilation
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        self.ram_layers = ModuleList()
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            # multi grid is applied to last layer only
            stage_multi_grid = multi_grid if i == len(
                self.stage_blocks) - 1 else None
            planes = base_channels * 2**i
            res_layer = make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=stage_plugins,
                multi_grid=stage_multi_grid,
                contract_dilation=contract_dilation,
                init_cfg=block_init_cfg)
            layer_name = f'layer{i+1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
            # ram layer初始化, in_channel与swin out_channel一致
            swin_channels = swin_channels * 2**i
            if is_res_ram:
                ram_layer = RamLayer(
                    in_channel=swin_channels,
                    out_channel=planes * 4,
                    is_res_ram=is_res_ram,
                    ram_simple=ram_simple)
                self.ram_layers.append(ram_layer)
            self.inplanes = planes * self.block.expansion

        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * 2**(
            len(self.stage_blocks) - 1)
        self.is_res_ram = is_res_ram

    def make_stage_plugins(self, plugins, stage_idx):
        """make plugins for ResNet 'stage_idx'th stage .

        Currently, we support to insert 'context_block',
        'empirical_attention_block', 'nonlocal_block' into the backbone like
        ResNet/ResNeXt. They could be inserted after conv1/conv2/conv3 of
        Bottleneck.

        An example of plugins format could be :
        >>> plugins=[
        ...     dict(cfg=dict(type='xxx', arg1='xxx'),
        ...          stages=(False, True, True, True),
        ...          position='after_conv2'),
        ...     dict(cfg=dict(type='yyy'),
        ...          stages=(True, True, True, True),
        ...          position='after_conv3'),
        ...     dict(cfg=dict(type='zzz', postfix='1'),
        ...          stages=(True, True, True, True),
        ...          position='after_conv3'),
        ...     dict(cfg=dict(type='zzz', postfix='2'),
        ...          stages=(True, True, True, True),
        ...          position='after_conv3')
        ... ]
        >>> self = ResNet(depth=18)
        >>> stage_plugins = self.make_stage_plugins(plugins, 0)
        >>> assert len(stage_plugins) == 3

        Suppose 'stage_idx=0', the structure of blocks in the stage would be:
            conv1-> conv2->conv3->yyy->zzz1->zzz2
        Suppose 'stage_idx=1', the structure of blocks in the stage would be:
            conv1-> conv2->xxx->conv3->yyy->zzz1->zzz2

        If stages is missing, the plugin would be applied to all stages.

        Args:
            plugins (list[dict]): List of plugins cfg to build. The postfix is
                required if multiple same type plugins are inserted.
            stage_idx (int): Index of stage to build

        Returns:
            list[dict]: Plugins for current stage
        """
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            # whether to insert plugin into current stage
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        """Make stem layer for ResNet."""
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        """Freeze stages param and norm stats."""
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x, swin_out=None):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            # ResNet stage
            x = res_layer(x)
            # 每一个resnet layer，后面跟一个ram layer
            if self.is_res_ram:
                # RAM流程
                x = self.ram_layers[i](x, swin_out[i])

            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
