_base_ = ['/home/ggm/GGM/mmrotate/mmdetection/configs/GGM/faster_rcnn_r50_fpn_1x_ali.py',
          '/home/ggm/GGM/mmrotate/mmdetection/configs/_base_/swa.py']

only_swa_training = True
swa_training = True
swa_load_from = '/home/ggm/GGM/mmrotate/mmdetection/workdir/fasterRCNN/epoch_12.pth'
swa_resume_from = None

# swa optimizer
swa_optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
# swa learning policy
swa_lr_config = dict(
    policy='cyclic',
    target_ratio=(1, 0.01),
    cyclic_times=12,
    step_ratio_up=0.0)
# total_epochs = 12
# max_epochs=12