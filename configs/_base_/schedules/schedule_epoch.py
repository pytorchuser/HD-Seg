# optimizer
optimizer = dict(
    # 优化器种类，更多细节可参考 https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/default_constructor.py
    type='SGD',
    # 优化器的学习率，参数的使用细节请参照对应的 PyTorch 文档
    lr=0.01,
    # 动量大小 (Momentum)
    momentum=0.9,
    # SGD 的权重衰减 (weight decay)
    weight_decay=0.0005)
optim_wrapper = dict(
    # 优化器包装器(Optimizer wrapper)为更新参数提供了一个公共接口
    type='OptimWrapper',
    # 用于更新模型参数的优化器(Optimizer)
    optimizer=optimizer,
    # 如果 'clip_grad' 不是None，它将是 ' torch.nn.utils.clip_grad' 的参数。
    clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        # 调度流程的策略，同样支持 Step, CosineAnnealing, Cyclic 等.
        # 请从 https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/lr_scheduler.py 参考 LrUpdater 的细节
        type='PolyLR',
        # 训练结束时的最小学习率
        eta_min=1e-4,
        # 多项式衰减 (polynomial decay) 的幂
        power=0.9,
        # 开始更新参数的时间步(step)
        begin=0,
        # 停止更新参数的时间步(step)
        end=240000,
        # 是否按照 epoch 计算训练时间
        by_epoch=False)
]
# training schedule by epoch
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=100, val_begin=10, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# 默认钩子(hook)配置
default_hooks = dict(
    # 记录迭代过程中花费的时间
    timer=dict(type='IterTimerHook'),
    # 从'Runner'的不同组件收集和写入日志, 每50次迭代打印一次日志
    logger=dict(type='LoggerHook', interval=46, log_metric_by_epoch=True),
    # 更新优化器中的一些超参数，例如学习率
    param_scheduler=dict(type='ParamSchedulerHook'),
    # save_best = ['acc', 'top', 'AR@', 'auc', 'precision', 'mAP', 'mDice', 'mIoU', 'mAcc', 'aAcc']
    # 定期保存检查点(checkpoint)
    checkpoint=dict(type='CheckpointHook',
                    by_epoch=True,
                    interval=5,
                    max_keep_ckpts=1,
                    save_last=False,
                    save_best=['mDice'],
                    rule='greater'),
    # 用于分布式训练的数据加载采样器
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # 验证结果可视化
    visualization=dict(type='SegVisualizationHook'))
