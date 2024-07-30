import vision2d


def create_model(cfg):
    if cfg.name == 'vision2d':
        return vision2d.RSNA24Model(model_name=cfg.modelname, in_c=cfg.channels, n_classes=cfg.nclasses)
    else:
        raise NotImplementedError