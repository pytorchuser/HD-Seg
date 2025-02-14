# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=True)
# runtime settings
# training by epochs
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(by_epoch=True, interval=2, max_keep_ckpts=1)
# save_best = ['acc', 'top', 'AR@', 'auc', 'precision', 'mAP', 'mDice', 'mIoU', 'mAcc', 'aAcc']
evaluation = dict(interval=2, metric=['mDice', 'mIoU'], pre_eval=True, save_best='mDice')
