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
