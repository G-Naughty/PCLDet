# Copyright (c) OpenMMLab. All rights reserved.
from .convex_giou_loss import BCConvexGIoULoss, ConvexGIoULoss
from .gaussian_dist_loss import GDLoss
from .gaussian_dist_loss_v1 import GDLoss_v1
from .kf_iou_loss import KFLoss
from .kld_reppoints_loss import KLDRepPointsLoss
from .contrastive_proxy_anchor_loss import SupConProxyAnchorLoss

__all__ = [
    'GDLoss', 'GDLoss_v1', 'KFLoss', 'ConvexGIoULoss', 'BCConvexGIoULoss', 'KLDRepPointsLoss',
    'SupConProxyAnchorLoss'
]
