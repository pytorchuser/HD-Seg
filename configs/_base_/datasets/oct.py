# dataset settings
# 数据集类型，这将被用来定义数据集
dataset_type = 'OCTDataset'
# 数据的根路径
data_root = '../data/OCT/new1'
# 训练时的裁剪大小
crop_size = (512, 512)
# 数据增广时，图像裁剪的大小
img_scale = (605, 700)
# 训练流程
train_pipeline = [
    # 从文件路径里加载图像
    dict(type='LoadImageFromFile'),
    # 对于当前图像，加载它的标注图像
    dict(type='LoadAnnotations',  reduce_zero_label=False),
    # 调整输入图像大小(resize)和其标注图像的数据增广流程
    dict(
        type='RandomResize',
        # 图像裁剪的大小
        scale=img_scale,
        # 数据增广的比例范围
        ratio_range=(0.5, 2.0),
        # 调整图像大小时是否保持纵横比
        keep_ratio=True),
    # 随机裁剪当前图像和其标注图像的数据增广流程
    dict(
        type='RandomCrop',
        # 随机裁剪的大小
        crop_size=crop_size,
        # 单个类别可以填充的最大区域的比
        cat_max_ratio=0.75),
    # 0.5的几率水平翻转
    dict(type='RandomFlip', prob=0.5),
    # 光学上使用一些方法扭曲当前图像和其注释，默认每个转换都有0.5的概率进行
    # 这些转换流程包括：改变亮度，对比形变，将颜色从BGR转换为HSV，改变饱和度，改变色调，将颜色从HSV转换为BGR
    dict(type='PhotoMetricDistortion'),
    # 打包用于语义分割的输入数据
    dict(type='PackSegInputs')
]
# 测试时单尺度数据增强
test_pipeline = [
    # 从文件路径里加载图像
    dict(type='LoadImageFromFile'),
    # 使用调整图像大小(resize)增强
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    # 在' Resize '之后添加标注图像，不需要做调整图像大小(resize)的数据变换
    # 加载数据集提供的语义分割标注
    dict(type='LoadAnnotations', reduce_zero_label=False),
    # 打包用于语义分割的输入数据
    dict(type='PackSegInputs')
]
# 图像缩放比例
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
# 测试时多尺度数据增强
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                # 多尺度调整图像大小(resize)增强
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                # 多尺度调整图像翻转增强
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ],
            # 加载数据集提供的语义分割标注
            [dict(type='LoadAnnotations')],
            # 打包用于语义分割的输入数据
            [dict(type='PackSegInputs')]
        ])
]
# 训练数据加载器(dataloader)的配置
train_dataloader = dict(
    # 每一个GPU的batch size大小
    batch_size=2,
    # 为每一个GPU预读取数据的进程个数
    num_workers=4,
    # 在一个epoch结束后关闭worker进程，可以加快训练速度
    persistent_workers=True,
    sampler=dict(
        type='InfiniteSampler',
        # 训练时进行随机洗牌(shuffle)
        shuffle=True),
    # 训练数据集配置
    dataset=dict(
        # 数据集类型，详见mmseg/datasets/
        type=dataset_type,
        # 数据集的根目录
        data_root=data_root,
        # 训练数据的前缀
        data_prefix=dict(
            # 原图目录
            img_path='images/training',
            # 分割图目录
            seg_map_path='annotations/training'),
        # 数据处理流程，它通过之前创建的train_pipeline传递
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(
        type='DefaultSampler',
        # 训练时不进行随机洗牌(shuffle)
        shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader
# 精度评估方法，我们在这里使用 IoUMetric 进行评估
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
