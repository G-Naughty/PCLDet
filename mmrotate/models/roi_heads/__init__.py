# Copyright (c) OpenMMLab. All rights reserved.
from .bbox_heads import (RotatedBBoxHead, RotatedConvFCBBoxHead, RotatedShared2FCBBoxHead,
                         ContrastRotatedConvFCBBoxHead)
from .gv_ratio_roi_head import GVRatioRoIHead
from .oriented_standard_roi_head import OrientedStandardRoIHead
from .roi_extractors import RotatedSingleRoIExtractor
from .roi_trans_roi_head import RoITransRoIHead
from .rotate_standard_roi_head import RotatedStandardRoIHead
from .contrastive.con_roi_trans_roi_head import ConRoITransRoIHead
from .contrastive.con_oriented_standard_roi_head import OBBContrastRoIHead
__all__ = [
    'RotatedBBoxHead', 'RotatedConvFCBBoxHead', 'RotatedShared2FCBBoxHead',
    'RotatedStandardRoIHead', 'RotatedSingleRoIExtractor',
    'OrientedStandardRoIHead', 'RoITransRoIHead', 'GVRatioRoIHead',
    'ContrastRotatedConvFCBBoxHead', 'ConRoITransRoIHead','OBBContrastRoIHead'
]