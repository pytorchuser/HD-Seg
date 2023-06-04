# model settings
# 分割框架通常使用 SyncBN
norm_cfg = dict(type='BN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
# 数据预处理的配置项，通常包括图像的归一化和增强
data_preprocessor = dict(
    # 数据预处理的类型
    type='SegDataPreProcessor',
    # 用于归一化输入图像的平均值
    mean=[123.675, 116.28, 103.53],
    # 用于归一化输入图像的标准差
    std=[58.395, 57.12, 57.375],
    # 是否将图像从 BGR 转为 RGB
    bgr_to_rgb=True,
    # 图像的填充值
    pad_val=0,
    # 'gt_seg_map'的填充值
    seg_pad_val=255)
model = dict(
    # 分割器(segmentor)的名字
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    # 加载使用 ImageNet 预训练的主干网络
    pretrained=None,
    backbone=dict(
        # 主干网络的类别，更多细节请参考 mmseg/models/backbones/swin.py
        # type='SwinTransformer',
        type='SIMSwinTransformer',
        # type='SIMSwinTransformerPlus',
        # 是否使用sim模块
        is_sim=False,
        is_fcm=False,
        is_res_ram=False,
        is_swin_ram=False,
        ram_simple=True,
        # 预训练时输入图像的大小，默认224
        pretrain_img_size=224,
        # 特征维度，默认96
        embed_dims=96,
        # 块大小，默认4
        patch_size=4,
        # 窗口大小，默认7
        window_size=7,
        # mlp比例，默认4
        mlp_ratio=4,
        # 每个swin Transformer stage的深度
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        # 每个状态(stage)产生的特征图输出的索引
        out_indices=(0, 1, 2, 3),
        # 如果为True，则为query、key、value添加一个可学习偏差
        qkv_bias=True,
        # 如果设置，覆盖默认的head_dim qk刻度** -0.5
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        # 随机深度率，默认0.1
        drop_path_rate=0.1,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        # 归一化层(norm layer)的配置项
        norm_cfg=backbone_norm_cfg),
    decode_head=dict(
        # 解码头(decode head)的类别。可用选项请参 mmseg/models/decode_heads
        type='UPerCustomHead',
        # 解码头的输入通道数
        in_channels=[96, 192, 384, 768],
        # 被选择特征图(feature map)的索引
        in_index=[0, 1, 2, 3],
        # 平均池化(avg pooling)的规模(scales)。
        pool_scales=(1, 2, 3, 6),
        # 解码头中间态(intermediate)的通道数
        channels=512,
        # 进入最后分类层(classification layer)之前的 dropout 比例
        dropout_ratio=0.1,
        # 分割前景的种类数目。
        num_classes=11,
        # 归一化层的配置项
        norm_cfg=norm_cfg,
        # 解码过程中调整大小(resize)的 align_corners 参数
        align_corners=False,
        # 解码头(decode_head)里的损失函数的配置项
        loss_decode=dict(
            # 分割时使用的损失函数的类别
            type='CrossEntropyLoss',
            # 分割时是否使用 sigmoid 激活
            use_sigmoid=False,
            # 解码头的损失权重
            loss_weight=1.0)),
    # DeepLab used this class weight for cityscapes
    # class_weight=[0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539]
    # 辅助头(auxiliary head)的种类。可用选项请参考 mmseg/models/decode_heads
    # auxiliary_head=dict(
    #     type='FCNHead',
    #     # 辅助头的输入通道数
    #     in_channels=384,
    #     # 被选择的特征图(feature map)的索引
    #     in_index=2,
    #     # 辅助头中间态(intermediate)的通道数
    #     channels=256,
    #     # FCNHead 里卷积(convs)的数目，辅助头中通常为1
    #     num_convs=1,
    #     # 在分类层(classification layer)之前是否连接(concat)输入和卷积的输出
    #     concat_input=False,
    #     # 进入最后分类层(classification layer)之前的 dropout 比例
    #     dropout_ratio=0.1,
    #     # 分割前景的种类数目。
    #     num_classes=11,
    #     # 归一化层的配置项
    #     norm_cfg=norm_cfg,
    #     # 解码过程中调整大小(resize)的 align_corners 参数
    #     align_corners=False,
    #     # 辅助头(auxiliary head)里的损失函数的配置项
    #     # loss_decode=dict(
    #     #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    #     loss_decode=[dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=0.4),
    #                  dict(type='DiceLoss', loss_name='loss_dice', loss_weight=0.4)]),
    # model training and testing settings
    # train_cfg 当前仅是一个占位符
    train_cfg=dict(),
    # 测试模式，可选参数为 'whole' 和 'slide'.
    # 'whole': 在整张图像上全卷积(fully-convolutional)测试。
    # 'slide': 在输入图像上做滑窗预测
    test_cfg=dict(mode='whole'))
