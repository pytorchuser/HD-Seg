# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=240000,
        by_epoch=False)
]
# training schedule by epoch
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=50, val_begin=10, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    # save_best = ['acc', 'top', 'AR@', 'auc', 'precision', 'mAP', 'mDice', 'mIoU', 'mAcc', 'aAcc']
    checkpoint=dict(type='CheckpointHook',
                    by_epoch=True,
                    interval=5,
                    max_keep_ckpts=1,
                    metric=['mDice', 'mIoU'],
                    save_best=['mIoU'],
                    rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
