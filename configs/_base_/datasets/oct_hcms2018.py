# dataset settings
dataset_type = 'OCTHCMS2018Dataset'
# training dataset root
# data_root = '../data/OCT_Manual_Delineations-2018_June_29(HCMS)/pad/train'
# testing dataset root
data_root = '../data/OCT_Manual_Delineations-2018_June_29(HCMS)/pad/test'
img_scale = (1024, 512)
crop_size = (512, 512)
# img_scale = (1024, 128)
# crop_size = (128, 128)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations',  reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=img_scale,
        ratio_range=(0.5, 2.0),
        keep_ratio=False),
    # dict(type='Pad', size=(1024, 512), padding_mode='edge'),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=False),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    # dict(type='Pad', size=(1024, 512), padding_mode='edge'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ],
            [dict(type='LoadAnnotations')],
            [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/training', seg_map_path='annotations/training'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=8,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mDice', 'mIoU'])
test_evaluator = val_evaluator
