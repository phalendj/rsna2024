import random
import numpy as np
import torch
import os

import models
import training


relative_directory = '/data/phalendj/kaggle/rsna2024'
image_directory = f'{relative_directory}/train_images/'


def set_directories(cfg):
    global relative_directory
    global image_directory
    relative_directory = cfg.relative_directory
    image_directory = cfg.relative_directory + '/' + cfg.image_directory


def set_random_seed(seed: int = 2222, deterministic: bool = False):
    """Set seeds"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = deterministic  # type: ignore