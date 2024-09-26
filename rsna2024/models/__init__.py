from . import vision2d
from . import unet
from . import tdcnn
from . import lstmcnn
from . import conv3d
from . import conv2p1d
from . import full_level_tdcnn
from . import magic_models

def create_model(cfg, fold):
    if cfg.name == 'vision2d':
        return vision2d.RSNA24Model(model_name=cfg.model_name, in_c=cfg.channels, n_classes=cfg.nclasses)
    elif cfg.name == 'magic':
        return magic_models.MagicModel(model_name=cfg.model_name, n_classes=cfg.nclasses, num_points=cfg.num_points, dropout=cfg.dropout, bottleneck_layers=cfg.bottleneck_layers, instance_layers=cfg.num_layers)
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
        return tdcnn.TDCNNModel(model_name=cfg.model_name, img_size=(512, 512), in_c=1, n_classes=cfg.nclasses, num_layers=cfg.num_layers, dropout=cfg.dropout)
    elif cfg.name == 'tdcnninstance':
        return tdcnn.TDCNNInstanceModel(model_name=cfg.model_name, img_size=(512, 512), in_c=1, n_classes=cfg.nclasses, num_layers=cfg.num_layers, dropout=cfg.dropout)
    elif cfg.name == 'tdcnnlevel':
        return tdcnn.TDCNNLevelModel(model_name=cfg.model_name, img_size=(64, 64), in_c=1, n_classes=cfg.nclasses, num_layers=cfg.num_layers, dropout=cfg.dropout)
    elif cfg.name == 'tdcnnlevelside':
        return tdcnn.TDCNNLevelSideModel(model_name=cfg.model_name, img_size=(64, 64), in_c=1, n_classes=cfg.nclasses, num_layers=cfg.num_layers, dropout=cfg.dropout)
    elif cfg.name == 'lstmcnn':
        return lstmcnn.LSTMCNNModel(model_name=cfg.model_name, img_size=(512, 512), in_c=1, n_classes=cfg.nclasses, num_layers=cfg.num_layers, dropout=cfg.dropout)
    elif cfg.name == 'lstmcnninstance':
        return lstmcnn.LSTMCNNInstanceModel(model_name=cfg.model_name, img_size=(512, 512), in_c=1, n_classes=cfg.nclasses, num_layers=cfg.num_layers, dropout=cfg.dropout)
    elif cfg.name == 'lstmcnnlevel':
        return lstmcnn.LSTMCNNLevelModel(model_name=cfg.model_name, img_size=(64, 64), in_c=1, n_classes=cfg.nclasses, num_layers=cfg.num_layers, dropout=cfg.dropout)
    elif cfg.name == 'lstmcnnlevelside':
        return lstmcnn.LSTMCNNLevelSideModel(model_name=cfg.model_name, img_size=(64, 64), in_c=1, n_classes=cfg.nclasses, num_layers=cfg.num_layers, dropout=cfg.dropout)
    elif cfg.name == 'tdcnn2':
        return tdcnn.TDCNNModel2(model_name=cfg.model_name, img_size=(512, 512), in_c=1, n_classes=cfg.nclasses, num_layers=cfg.num_layers, dropout=cfg.dropout)
    elif cfg.name == 'tdcnninstance2':
        return tdcnn.TDCNNInstanceModel2(model_name=cfg.model_name, img_size=(512, 512), in_c=1, n_classes=cfg.nclasses, num_layers=cfg.num_layers, dropout=cfg.dropout)
    elif cfg.name == 'tdcnnlevel2':
        return tdcnn.TDCNNLevelModel2(model_name=cfg.model_name, img_size=(224, 224), in_c=1, n_classes=cfg.nclasses, num_layers=cfg.num_layers, dropout=cfg.dropout)
    elif cfg.name == 'tdcnnlevelside2':
        return tdcnn.TDCNNLevelSideModel2(model_name=cfg.model_name, img_size=(224, 224), in_c=1, n_classes=cfg.nclasses, num_layers=cfg.num_layers, dropout=cfg.dropout)
    elif cfg.name == 'tdcnn3':
        return tdcnn.TDCNNModel3(model_name=cfg.model_name, img_size=(512, 512), in_c=1, n_classes=cfg.nclasses, num_layers=cfg.num_layers, dropout=cfg.dropout)
    elif cfg.name == 'tdcnninstance3':
        return tdcnn.TDCNNInstanceModel3(model_name=cfg.model_name, img_size=(512, 512), in_c=1, n_classes=cfg.nclasses, num_layers=cfg.num_layers, dropout=cfg.dropout)
    elif cfg.name == 'tdcnnlevel3':
        return tdcnn.TDCNNLevelModel3(model_name=cfg.model_name, img_size=(224, 224), in_c=1, n_classes=cfg.nclasses, num_layers=cfg.num_layers, dropout=cfg.dropout)
    elif cfg.name == 'tdcnnlevelside3':
        return tdcnn.TDCNNLevelSideModel3(model_name=cfg.model_name, img_size=(224, 224), in_c=1, n_classes=cfg.nclasses, num_layers=cfg.num_layers, dropout=cfg.dropout)
    
    elif cfg.name == 'tdcnnalllevelside2':
        model = tdcnn.TDCNNAllLevelSideModel2(model_name=cfg.model_name, img_size=(224, 224), hidden_dim=cfg.hidden_dim,
                                             in_c=1, n_classes=cfg.nclasses, num_layers=cfg.num_layers, dropout=cfg.dropout)
        model.load(cfg.preload, fold=fold)
        return model

    elif cfg.name == 'tdcnnallxlevelside2':
        model = tdcnn.TDCNNAllXLevelSideModel2(model_name=cfg.model_name, img_size=(224, 224), hidden_dim=cfg.hidden_dim,
                                             in_c=1, n_classes=cfg.nclasses, num_layers=cfg.num_layers, dropout=cfg.dropout)
        model.load(cfg.preload, fold=fold)
        return model


    elif cfg.name == 'fusedtdcnnlevel':
        model = tdcnn.FusedTDCNNLevelModel(model_name=cfg.model_name, 
                                          sagittal_t2_model=cfg.sagittal_t2,
                                          sagittal_t1_model=cfg.sagittal_t1,
                                          axial_t2_model=cfg.axial_t2,
                                          fold=fold,
                                          img_size=(64, 64), in_c=1, n_classes=cfg.nclasses, num_layers=cfg.num_layers)
        model.freeze_vision()
        return model

    elif cfg.name == 'tdcnnunetpreloadzoom':
        return tdcnn.TDCNNUNetPreloadZoom(in_channels=cfg.channels, out_classes=cfg.unet_classes, patch_size=cfg.patch_size, encoder_name=cfg.encodername, 
                                          classifier_name=cfg.model_name, classifier_classes=cfg.nclasses, subsize=cfg.subsize, load_dir=cfg.preload, fold=fold)
    elif cfg.name == 'doubletdcnnunetpreloadzoom':
        return tdcnn.DoubleTDCNNUNetPreloadZoom(in_channels=cfg.channels, out_classes=cfg.unet_classes, patch_size=cfg.patch_size, encoder_name=cfg.encodername, 
                                                classifier_name=cfg.model_name, classifier_classes=cfg.nclasses, subsize=cfg.subsize, load_dir=cfg.preload, fold=fold, 
                                                condition=cfg.condition)
    
    elif cfg.name == 'ResNet3D18':
        return conv3d.ResNet3D18(cfg.nclasses)
    elif cfg.name == 'ResNet2p1D18':
        return conv2p1d.ResNet2p1D18(cfg.nclasses)
    elif cfg.name == 'FullLevelTDCNN':
        if cfg.preload is not None:
            model = full_level_tdcnn.FullLevelTDCNNModel(model_name=cfg.model_name, n_classes=cfg.nclasses, img_size=(512, 512), use=cfg.use, num_layers=cfg.num_layers)
            model.load(cfg.preload, fold=fold)
            return model
        else:
            return full_level_tdcnn.FullLevelTDCNNModel(model_name=cfg.model_name, n_classes=cfg.nclasses, img_size=(512, 512), use=cfg.use, num_layers=cfg.num_layers)
    else:
        raise NotImplementedError