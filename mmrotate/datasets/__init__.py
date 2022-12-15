# Copyright (c) OpenMMLab. All rights reserved.
from .builder import build_dataset  # noqa: F401, F403
from .dota import DOTADataset  # noqa: F401, F403
from .hrsc import HRSCDataset  # noqa: F401, F403
from .pipelines import *  # noqa: F401, F403
from .sar import SARDataset  # noqa: F401, F403
from .fair1m import Fair1mDataset
from .rubbish import RubbishDataset
from .hfsd import HFSDDataset
__all__ = ['SARDataset', 'DOTADataset', 'build_dataset', 'HRSCDataset',
           'Fair1mDataset', 'RubbishDataset', 'HFSDDataset']
