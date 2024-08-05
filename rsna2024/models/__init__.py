from . import vision2d
from . import unet
from . import tdcnn

def create_model(cfg, fold):
    if cfg.name == 'vision2d':
        return vision2d.RSNA24Model(model_name=cfg.modelname, in_c=cfg.channels, n_classes=cfg.nclasses)
    elif cfg.name == 'unet':
        return unet.UNet(in_channels=cfg.channels, out_classes=cfg.unet_classes, patch_size=cfg.patch_size, encoder_name=cfg.encodername, 
                         classifier_name=cfg.model_name, classifier_classes=cfg.nclasses)
    elif cfg.name == 'unetpreload':
        return unet.UNetPreload(in_channels=cfg.channels, out_classes=cfg.unet_classes, patch_size=cfg.patch_size, encoder_name=cfg.encodername, 
                         classifier_name=cfg.model_name, classifier_classes=cfg.nclasses, load_dir=cfg.preload, fold=fold)
    elif cfg.name == 'unetpreloadzoom':
        return unet.UNetPreloadZoom(in_channels=cfg.channels, out_classes=cfg.unet_classes, patch_size=cfg.patch_size, encoder_name=cfg.encodername, 
                         classifier_name=cfg.model_name, classifier_classes=cfg.nclasses, subsize=cfg.subsize, load_dir=cfg.preload, fold=fold)
    elif cfg.name == 'tdcnn':
        return tdcnn.TDCNNModel(model_name=cfg.model_name, img_size=(512, 512), in_c=1, n_classes=cfg.nclasses, num_layers=cfg.num_layers, pretrained=True)
    elif cfg.name == 'tdcnnlevel':
        return tdcnn.TDCNNLevelModel(model_name=cfg.model_name, img_size=(64, 64), in_c=1, n_classes=cfg.nclasses, num_layers=cfg.num_layers, pretrained=True)
    
    elif cfg.name == 'fusedtdcnnlevel':
        return tdcnn.FusedTDCNNLevelModel(model_name=cfg.model_name, img_size=(64, 64), in_c=1, n_classes=cfg.nclasses, num_layers=cfg.num_layers, pretrained=True)

    elif cfg.name == 'tdcnnunetpreloadzoom':
        return tdcnn.TDCNNUNetPreloadZoom(in_channels=cfg.channels, out_classes=cfg.unet_classes, patch_size=cfg.patch_size, encoder_name=cfg.encodername, 
                                          classifier_name=cfg.model_name, classifier_classes=cfg.nclasses, subsize=cfg.subsize, load_dir=cfg.preload, fold=fold)
    elif cfg.name == 'doubletdcnnunetpreloadzoom':
        return tdcnn.DoubleTDCNNUNetPreloadZoom(in_channels=cfg.channels, out_classes=cfg.unet_classes, patch_size=cfg.patch_size, encoder_name=cfg.encodername, 
                                                classifier_name=cfg.model_name, classifier_classes=cfg.nclasses, subsize=cfg.subsize, load_dir=cfg.preload, fold=fold, 
                                                condition=cfg.condition)
    else:
        raise NotImplementedError