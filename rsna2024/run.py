import hydra
import logging
import datetime
from omegaconf import DictConfig

try:
    from .utils import set_directories, set_random_seed
except ImportError:
    from utils import set_directories, set_random_seed

import models
import training


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config1")
def run(cfg: DictConfig) -> None:
    logging.info(f'Configuration: {cfg}')
    set_random_seed(cfg.seed)
    set_directories(cfg.directories)
    # torch.set_float32_matmul_precision('high')
    for fold in cfg.training.use_folds:
        logger.info(f'Run Fold {fold}')
        model = models.create_model(cfg=cfg.model)
        training.train_one_fold(model=model, cfg=cfg, fold=fold)

    model = models.create_model(cfg=cfg.model)
    training.evaluate(model, cfg=cfg)


if __name__ == '__main__':
    log_filename = f"rsna{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.log"
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s',
                        handlers=[logging.FileHandler(log_filename, mode='w'),
                                  logging.StreamHandler()])
    
    run()
