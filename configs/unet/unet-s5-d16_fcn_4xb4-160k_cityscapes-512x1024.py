_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/oct_hcms2018.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_epoch.py'
]
load_from = '../pth/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_20211210_145204-6860854e.pth'  # noqa
NUM_CLASSES = 9

data_preprocessor = dict(size=(512, 512))

model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=NUM_CLASSES, loss_decode=[
        dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
        dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0)
    ]),
    auxiliary_head=dict(num_classes=NUM_CLASSES)
)

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(
     _delete_=True,
     type='AdamW',
     lr=0.00018,
     betas=(0.9, 0.999),
     weight_decay=0.01,
     )
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
        end=50,
        gamma=0.8,
        by_epoch=True)]
train_dataloader = dict(
    batch_size=4
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
