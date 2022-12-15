_base_ = ['../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py', '../_base_/swa.py']

# swa optimizer
swa_optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
##paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))