import hydra
import logging
import datetime
from omegaconf import DictConfig

try:
    from .utils import set_directories, set_random_seed, set_clean, set_debug, set_preload
except ImportError:
    from utils import set_directories, set_random_seed, set_clean, set_debug, set_preload

import models
import training
import training.evaluate as evaluation


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config1")
def run(cfg: DictConfig) -> None:
    logging.info(f'Configuration: {cfg}')
    set_clean(cfg.clean)
    set_debug(cfg.debug)
    set_random_seed(cfg.seed, deterministic=True)
    set_directories(cfg.directories)
    set_preload(cfg.mode.lower() != 'test')
    # torch.set_float32_matmul_precision('high')
    if cfg.train:
        for fold in cfg.training.use_folds:
            logger.info(f'Run Fold {fold}')
            model = models.create_model(cfg=cfg.model, fold=fold)
            training.train_one_fold(model=model, cfg=cfg, fold=fold)

    if cfg.result == 'evaluate':
        model = models.create_model(cfg=cfg.model, fold=0)
        evaluation.evaluate(model, cfg=cfg)
    elif cfg.result == 'instance_numbers':
        evaluation.generate_instance_numbers(cfg)
    elif cfg.result == 'sagittalcenters':
        evaluation.generate_xy_values(cfg=cfg)
    elif cfg.result == 'predict':
        evaluation.predict(cfg=cfg)


if __name__ == '__main__':
    log_filename = f"rsna{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.log"
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s',
                        handlers=[logging.FileHandler(log_filename, mode='w'),
                                  logging.StreamHandler()])
    
    run()
