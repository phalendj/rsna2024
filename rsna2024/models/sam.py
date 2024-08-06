import torch
import logging
from pathlib import Path
import torch.nn as nn
import segmentation_models_pytorch as smp
import numpy as np
from . import vision2d

logger = logging.getLogger(__name__)


## TODO:
# https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/524194
# https://github.com/MedicineToken/Medical-SAM2