import timm
import logging
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F


from . import unet


logger = logging.getLogger(__name__)


class TDCNNModel(nn.Module):
    def __init__(self, model_name: str, img_size: tuple[int, int], in_c: int = 1, n_classes: int = 3, num_layers: int = 4, pretrained: bool = True):
        super().__init__()
        self.feature_model = timm.create_model(
                                    model_name,
                                    pretrained=pretrained, 
                                    features_only=False,
                                    in_chans=in_c,
                                    num_classes=n_classes,
                                    global_pool='avg'
                                    )
        X = torch.randn(2, 1, *img_size)
        Y = self.feature_model.forward_features(X)
        d_model = Y.shape[1]
        print(f'Feature dimension {d_model}')
        
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.LazyLinear(n_classes)

        self.model_name = model_name
    
    def load_feature_model(self, fname):
        self.feature_model.load_state_dict(torch.load(fname))
        
    def freeze_features(self):
        print('freeze features')
        for parameters in self.feature_model.parameters():
            parameters.requires_grad = False
        
    def unfreeze_features(self):
        print('unfreeze features')
        for parameters in self.feature_model.parameters():
            parameters.requires_grad = True

    def name(self):
        return f'td_cnn_{self.model_name}'

    def forward_features(self, x):
        B, I, H, W = x.shape
        x = x.unsqueeze(2).flatten(0,1)
        y = self.feature_model.forward_features(x)  # B*I, D, 4 ,4
        y = self.pool(y).flatten(1)  # B*I, D
        y = y.reshape(B, I, -1)
        y = self.encoder(y)  # B, I, 1024
        #TODO: Try different pooling methods: avg, max, catavgmax
        y = F.adaptive_avg_pool1d(y.transpose(-1, -2), 1).squeeze(-1)  # B, 1024
        return y

    def forward(self, x):
        y = self.forward_features(x)
        return {'labels': self.classifier(y)}
    
class TDCNNLevelModel(TDCNNModel):
    def level_forward(self, x):
        B, L, I, H, W = x.shape
        x = x.flatten(0, 1)  # Now first index is is B0L0, B0L1, ..., B0L4, B1L0, B1L1, ... , dim = (B*L, I, H, W)
        t = super().forward(x)
        y = t['labels']
        # y.shape = B*L, nclasses
        y = y.reshape(B, L, -1, 3)  # Now B, Condition, Level, diagnosis
        y = y.transpose(1,2).flatten(1)  # Now B, nclasses*nlevels
        return y
    
    def level_forward_features(self, x):
        B, L, I, H, W = x.shape
        x = x.flatten(0, 1)  # Now first index is is B0L0, B0L1, ..., B0L4, B1L0, B1L1, ... , dim = (B*L, I, H, W)
        y = super().forward_features(x)
        # y.shape = B*L, feature dim (1024 or something like that)
        y = y.reshape(B, L, -1)  # Now B, Level, feature dim
        return y

    def name(self):
        return f'td_cnn_level_{self.model_name}'

    def load(self, load_dir, fold):
        fname = Path(load_dir) / (self.name() + f'_fold{fold}.pth')
        logger.info(f'Loading Model from {fname}')
        self.load_state_dict(torch.load(fname))

    def freeze_vision(self):
        for parameters in self.feature_model.parameters():
            parameters.requires_grad = False

    def unfreeze_vision(self):
        for parameters in self.feature_model.parameters():
            parameters.requires_grad = True

    def forward(self, x):
        y = self.level_forward(x)
        return {'labels': y}
    

class FusedTDCNNLevelModel(nn.Module):
    def __init__(self, model_name: str, 
                 sagittal_t2_model: str,
                 sagittal_t1_model: str,
                 axial_t2_model: str,
                 fold: int,
                 img_size: tuple[int, int], 
                 in_c: int = 1, 
                 n_classes: int = 3, 
                 num_layers: int = 4
                 ):
        super().__init__()
        self.sagittal_t2 = TDCNNLevelModel(model_name=model_name, img_size=img_size, in_c=in_c, n_classes=n_classes, num_layers=num_layers)
        self.sagittal_t1 = TDCNNLevelModel(model_name=model_name, img_size=img_size, in_c=in_c, n_classes=n_classes, num_layers=num_layers)
        self.axial_t2 = TDCNNLevelModel(model_name=model_name, img_size=img_size, in_c=in_c, n_classes=n_classes, num_layers=num_layers)

        self.sagittal_t2.load(sagittal_t2_model, fold)
        self.sagittal_t1.load(sagittal_t1_model, fold)
        self.axial_t2.load(axial_t2_model, fold)

        self.classifier = nn.LazyLinear(n_classes)

        self.model_name = model_name

    def freeze_vision(self):
        logger.info('Freeze Vision model')
        self.sagittal_t2.freeze_vision()
        self.sagittal_t1.freeze_vision()
        self.axial_t2.freeze_vision()
        
    def unfreeze_vision(self):
        logger.info('Unfreeze Vision model')
        self.sagittal_t2.unfreeze_vision()
        self.sagittal_t1.unfreeze_vision()
        self.axial_t2.unfreeze_vision()

    def name(self):
        return f'fused_td_cnn_level_{self.model_name}'

    def forward(self, x1, x2, x3):
        B, L, I, H, W = x1.shape
        y1 = self.sagittal_t2.level_forward_features(x1)
        y2 = self.sagittal_t1.level_forward_features(x2)
        y3 = self.axial_t2.level_forward_features(x3)

        y = torch.concat([y1, y2, y3], dim=2)
        y = self.classifier(y)
        y = y.reshape(B, L, -1, 3)  # Now B, Condition, Level, diagnosis
        y = y.transpose(1,2).flatten(1)  # Now B, nclasses*nlevels
        return {'labels': y}


class TDCNNUNetPreloadZoom(nn.Module):
    def __init__(self, in_channels, out_classes, patch_size, encoder_name, classifier_name, classifier_classes, subsize: int, load_dir: str, fold: int, predict_classes: int = 3):
        super(TDCNNUNetPreloadZoom, self).__init__()
        self.unet_in_channels = in_channels
        self.UNet = unet.UNetPreload(in_channels, out_classes, patch_size, encoder_name, classifier_name, classifier_classes, load_dir, fold)
        
        self.subsize = subsize
        self.predict_classes = predict_classes
        self.classifier = TDCNNModel(model_name=classifier_name, img_size=(subsize, subsize), in_c=1, n_classes=predict_classes)

    def name(self):
        return f'tdcnn_unetzoom_{self.UNet.encoder_name}_{self.classifier.name()}'

    def get_zoom(self, mask, x):
        coord = torch.argwhere(mask > 0.9).float().mean(dim=0).long()
        if self.training:
            coord[0] += np.random.randint(-10, 11)
            coord[1] += np.random.randint(-10, 11)
        X, Y = torch.clip(coord, self.subsize, self.UNet.patch_size - self.subsize)
        return x[:, (X-self.subsize):(X+self.subsize), (Y-self.subsize):(Y+self.subsize)]

    def forward(self,X):
        # X is B, I, H, W, need to trim to middle channels for UNet
        i0 = (X.shape[1] - self.unet_in_channels) // 2

        y = self.UNet(X[:, i0:i0+self.unet_in_channels])
        masks = y['masks']
        B = X.shape[0]
        M = masks.shape[1]

        X2 = torch.stack(sum([[self.get_zoom(masks[i, j], X[i]) for j in range(M)] for i in range(B)],[]), dim=0)  # reshapes batch index to B*Mask number, which is number of levels
        pred2 = self.classifier(X2)

        y['labels'] = pred2['labels'].reshape(B, -1)

        return y
    

class DoubleTDCNNUNetPreloadZoom(nn.Module):
    def __init__(self, in_channels, out_classes, patch_size, encoder_name, classifier_name, classifier_classes, subsize: int, load_dir: str, fold: int, condition: str = 'spinal'):
        super(DoubleTDCNNUNetPreloadZoom, self).__init__()
        self.unet_in_channels = in_channels
        self.UNet = unet.UNetPreload(in_channels, out_classes, patch_size, encoder_name, classifier_name, classifier_classes, load_dir, fold)
        
        self.subsize = subsize
        self.condition = condition
        self.predict_classes = 3 if condition == 'spinal' else 6
        self.classifier = TDCNNModel(model_name=classifier_name, img_size=(subsize, subsize), in_c=1, n_classes=self.predict_classes)

    def name(self):
        return f'tdcnn_unetzoom_{self.UNet.encoder_name}_{self.classifier.name()}'

    def get_zoom(self, mask, x):
        coord = torch.argwhere(mask > 0.9).float().mean(dim=0).long()
        if self.training:
            coord[0] += np.random.randint(-10, 11)
            coord[1] += np.random.randint(-10, 11)
        X, Y = torch.clip(coord, self.subsize, self.UNet.patch_size - self.subsize)
        return x[:, (X-self.subsize):(X+self.subsize), (Y-self.subsize):(Y+self.subsize)]

    def forward(self,X):
        # X is B, I, H, W, need to trim to middle channels for UNet
        i0 = (X.shape[1] - self.unet_in_channels) // 2

        y = self.UNet(X[:, i0:i0+self.unet_in_channels])
        masks = y['masks']
        B = X.shape[0]
        M = masks.shape[1]

        X2 = torch.stack(sum([[self.get_zoom(masks[i, j], X[i]) for j in range(M)] for i in range(B)],[]), dim=0)  # reshapes batch index to B*Mask number, which is number of levels
        pred2 = self.classifier(X2)

        if self.condition == 'spinal':
            y['labels'] = pred2['labels'].reshape(B, -1)
        elif self.condition == 'foraminal':
            t = pred2['labels'].reshape(B, -1, self.predict_classes)
            t = (t[:, :5, :] + t[:, 5:, :]) * 0.5
            y['labels'] = torch.concat([t[:, :, :3].flatten(1), t[:, :, 3:].flatten(1)], dim=1)
        else:
            raise NotImplementedError


        return y