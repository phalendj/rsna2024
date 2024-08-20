import random
import numpy as np
import torch
import os
import logging


logger = logging.getLogger(__name__)


relative_directory = '/data/phalendj/kaggle/rsna2024'
image_directory = f'{relative_directory}/train_images/'

CLEAN = True
DEBUG = False
PRELOAD = True

def set_directories(cfg):
    global relative_directory
    global image_directory
    logger.info(f'Set directoreis to {cfg.relative_directory} / {cfg.image_directory}')
    relative_directory = cfg.relative_directory
    image_directory = cfg.relative_directory + '/' + cfg.image_directory + '/'


def set_random_seed(seed: int = 2222, deterministic: bool = False):
    """Set seeds"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = deterministic  # type: ignore


def set_clean(val):
    global CLEAN
    CLEAN = val

def set_debug(val):
    global DEBUG
    DEBUG = val

def set_preload(val):
    global PRELOAD
    PRELOAD = val


def in_notebook() -> bool:
    """Used for switching display modes based on if in notebook or not"""
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except (ImportError, AttributeError):
        return False
    return True
