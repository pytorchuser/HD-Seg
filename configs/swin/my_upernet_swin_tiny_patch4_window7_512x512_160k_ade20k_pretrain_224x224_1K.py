_base_ = [
    '../_base_/models/upernet_custom_swin.py', '../_base_/datasets/oct_hcms2018.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_epoch.py'
]
load_from = '../pth/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210531_112542-e380ad3e.pth'  # noqa
# load_from = '../pth/oct_T_8_lr_h_atdr&ddr=0.2_keep_ratio=False_octhcms2018org_epoch100_ppm_1(3)_epoch_82.pth'  # noqa
NUM_CLASSES = 9

data_preprocessor = dict(size=(512, 512))

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        qkv_bias=True,
        qk_scale=True,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        attn_drop_rate=0.3,
        patch_norm=True),
    decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=NUM_CLASSES, dropout_ratio=0.2,
                     loss_decode=[dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
                     dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)],
                     # TODO 此处添加配置信息msc_module_cfg
                     # msc_module_cfg=[
                     #     dict(type='PPM', layer_idx=0), dict(type='PPM', layer_idx=1),
                     #     dict(type='PPM', layer_idx=2), dict(type='PPM', layer_idx=3)]
                     msc_module_cfg=[
                        dict(type='UFE', layer_idx=0, ufe_cfg=dict(
                            num_stages=4,
                            strides=(1, 1, 1, 1),
                            enc_num_convs=(1, 1, 1, 1),
                            dec_num_convs=(1, 1, 1),
                            downsamples=(True, True, True),
                            enc_dilations=(1, 1, 1, 1),
                            dec_dilations=(1, 1, 1),)),
                        dict(type='UFE', layer_idx=1, ufe_cfg=dict(
                            num_stages=3,
                            strides=(1, 1, 1),
                            enc_num_convs=(1, 1, 1),
                            dec_num_convs=(1, 1),
                            downsamples=(True, True),
                            enc_dilations=(1, 1, 1),
                            dec_dilations=(1, 1),)),
                        dict(type='UFE', layer_idx=2, ufe_cfg=dict(
                            num_stages=2,
                            strides=(1, 1),
                            enc_num_convs=(1, 1),
                            dec_num_convs=([1]),
                            downsamples=([True]),
                            enc_dilations=(1, 1),
                            dec_dilations=([1]),)),
                        dict(type='PPM', layer_idx=3)
                     ]
                     # msc_module_cfg=[
                     #     dict(type='PPM', layer_idx=0), dict(type='PPM', layer_idx=1)]
                     # msc_module_cfg=[
                     #     dict(type='PPM', layer_idx=1), dict(type='PPM', layer_idx=2),
                     #     dict(type='PPM', layer_idx=3)]
                     ),
    auxiliary_head=dict(in_channels=384, dropout_ratio=0.1, num_classes=NUM_CLASSES))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(
     _delete_=True,
     type='AdamW',
     lr=0.00018,
     betas=(0.9, 0.999),
     weight_decay=0.01,
     )

# optimizer = dict(
#     # 优化器种类，更多细节可参考 https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/default_constructor.py
#     type='SGD',
#     # 优化器的学习率，参数的使用细节请参照对应的 PyTorch 文档
#     lr=0.01,
#     # 动量大小 (Momentum)
#     momentum=0.9,
#     # SGD 的权重衰减 (weight decay)
#     weight_decay=0.0005)

optim_wrapper = dict(
    # 优化器包装器(Optimizer wrapper)为更新参数提供了一个公共接口
    type='AmpOptimWrapper',
    # 用于更新模型参数的优化器(Optimizer)
    optimizer=optimizer,
    # 如果 'clip_grad' 不是None，它将是 ' torch.nn.utils.clip_grad' 的参数。
    clip_grad=None,
    paramwise_cfg=dict(
        # 优化器配置去重
        bypass_duplicate=True,
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        })
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        end_factor=1,
        by_epoch=True,
        begin=0,
        end=10),
    # dict(
    #     type='PolyParamScheduler',
    #     param_name='lr',
    #     eta_min=0,
    #     begin=15,
    #     end=50,
    #     power=1.0,
    #     by_epoch=True)
    dict(
        type='StepParamScheduler',
        param_name='lr',
        step_size=7,
        begin=10,
        end=100,
        gamma=0.8,
        by_epoch=True)
    # dict(
    #     type='CosineAnnealingParamScheduler',
    #     # 需要调整的参数名称，如lr、momentum等。
    #     param_name='lr',
    #     eta_min=1e-7,
    #     T_max=40,
    #     by_epoch=True,
    #     begin=10,
    #     end=50,
    #     # 是否为每次更新打印值。默认为False。
    #     verbose=False)
    # dict(
    #     type='OneCycleParamScheduler',
    #     param_name='lr',
    #     eta_max=0.00006,
    #     total_steps=50,
    #     # 学习率上升所占的比例
    #     pct_start=0.3,
    #     # 指定退火策略:“cos”表示余弦退火，“linear”表示线性退火。默认为' cos '
    #     anneal_strategy='cos',
    #     # 通过initial_param = eta_max/div_factor决定初始学习率,默认为25
    #     div_factor=25,
    #     # 通过eta_min = initial_param/final_div_factor确定最小学习率,默认为1e4
    #     final_div_factor=1e4,
    #     three_phase=False,
    #     by_epoch=True)
    # dict(
    #     type='ReduceOnPlateauParamScheduler',
    #     # 需要调整的参数名称，如lr、momentum等。
    #     param_name='lr',
    #     # monitor是度量模型的性能是否得到改进的度量标准的名称。
    #     monitor='mDice',
    #     # 在less规则中，当monitor停止减少时，参数将减少;在greater规则下，当monitor停止增加时，参数将将减少。
    #     rule='greater',
    #     # factor是每次学习率下降的比例，新的学习率等于老的学习率乘以factor。
    #     factor=0.5,
    #     # patience是能够容忍的次数，当patience次后，网络性能仍未提升，则会降低学习率。
    #     patience=5,
    #     # threshold是测量新的最优的阈值，一般只关注相对大的性能提升。
    #     threshold=1e-4,
    #     # One of rel, abs. In rel rule, dynamic_threshold = best * ( 1 + threshold ) in ‘greater’ rule or
    #     # best * ( 1 - threshold ) in less rule. In abs rule, dynamic_threshold = best + threshold in greater rule or
    #     # best - threshold in less rule. Defaults to ‘rel’.
    #     threshold_rule='rel',
    #     # 参数减少后恢复正常操作所需等待的epoch数。默认为0。
    #     cooldown=0,
    #     # 各参数组参数的下界。默认为0。
    #     min_value=0,
    #     # eps指最小的学习率变化，当新旧学习率差别小于eps时，维持学习率不变。
    #     eps=1e-8,
    #     begin=10,
    #     end=50,
    #     by_epoch=True,
    #     # 是否为每次更新打印值。默认为False。
    #     verbose=False)
]

# By default, models are trained on 8 GPUs with 2 images per GPU
# CUDA out of memory。 加载验证数据集时内存会爆掉， workers_per_gpu设置的小一些可避免这个问题
train_dataloader = dict(
    batch_size=8
)
val_dataloader = dict(
    batch_size=8,
    num_workers=1,
    sampler=dict(
        type='DefaultSampler',
        # 训练时进行随机洗牌(shuffle)
        shuffle=True)
)
test_dataloader = dict(
    batch_size=8,
    num_workers=1,
    sampler=dict(
        type='DefaultSampler',
        # 训练时进行随机洗牌(shuffle)
        shuffle=True)
)
