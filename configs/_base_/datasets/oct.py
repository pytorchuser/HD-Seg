# dataset settings
dataset_type = 'OCTDataset'
data_root = '../data/OCT/new1'
img_norm_cfg = dict(
    # mean均值， std标准差，三通道顺序为rgb，以下为pytorch的基模型，如果要从头训练一个数据集，可以自行计算
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# 不对应(h,w)属性，指最长边和最短边。
img_scale = (605, 700)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations',  reduce_zero_label=False),
    # 变化图像和其注释大小的数据增广的流程,图像的最大规模,缩放图像的比例范围
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0), keep_ratio=True),
    # 实现随机裁剪，随机裁剪图像生成 patch 的大小，单个类别可以填充的最大区域的比例
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    # 0.5的几率水平翻转
    dict(type='RandomFlip', prob=0.5),
    # 光学上使用一些方法扭曲当前图像和其注释，默认每个转换都有0.5的概率进行
    # 这些转换流程包括：改变亮度，对比形变，将颜色从BGR转换为HSV，改变饱和度，改变色调，将颜色从HSV转换为BGR
    dict(type='PhotoMetricDistortion'),
    # 归一化
    dict(type='Normalize', **img_norm_cfg),
    # 填充当前图像到指定大小的数据，填充的图像大小，图像的填充值，'gt_semantic_seg'的填充值
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    # 流程里收集数据的默认格式捆，将 'img' 和 'gt_semantic_seg'装入DataContainer
    dict(type='DefaultFormatBundle'),
    # 决定数据里哪些键被传递到分割器里的流程
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
# 创建验证数据集时，使用test_pipeline作为compose参数
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=40000,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='images/training',
            ann_dir='annotations/training',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline))
