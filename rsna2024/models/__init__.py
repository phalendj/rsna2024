from . import vision2d
from . import unet

def create_model(cfg):
    if cfg.name == 'vision2d':
        return vision2d.RSNA24Model(model_name=cfg.modelname, in_c=cfg.channels, n_classes=cfg.nclasses)
    if cfg.name == 'unet':
        return unet.UNet(in_channels=cfg.channels, out_classes=cfg.unet_classes, patch_size=cfg.patch_size, encoder_name=cfg.encodername, 
                         classifier_name=cfg.model_name, classifier_classes=cfg.nclasses)
    else:
        raise NotImplementedError