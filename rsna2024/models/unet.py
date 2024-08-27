import torch
import logging
from pathlib import Path
import torch.nn as nn
import segmentation_models_pytorch as smp
import numpy as np
from . import vision2d
try:
    from .. import utils as rsnautils
except ImportError:
    import utils as rsnautils

logger = logging.getLogger(__name__)

# Can also use https://www.kaggle.com/datasets/tabassumnova/lumbar-spine-segmentation to pretrain



class UNet(nn.Module):
    def __init__(self, in_channels, out_classes, patch_size, encoder_name, classifier_name, classifier_classes):
        super(UNet, self).__init__()
        self.patch_size = patch_size
        self.UNet = smp.Unet(
            encoder_name=encoder_name,
            classes=out_classes,
            in_channels=in_channels,
            encoder_weights='imagenet' if rsnautils.PRELOAD else None
        )
        self.encoder_name = encoder_name
        self.out_classes = out_classes

        final_channels = out_classes + in_channels
        self.classifier_classes = classifier_classes
        self.classifier = vision2d.RSNA24Model(model_name=classifier_name, in_c=final_channels, n_classes=classifier_classes)

    def name(self):
        return f'unet_{self.encoder_name}_{self.classifier.name()}'

    def forward(self,X):
        if isinstance(X, dict):
            key = [k for k in X.keys() if 'Patch' in k][0]
            X = X[key]
        x = self.UNet(X)
#       MinMaxScaling along the class plane to generate a heatmap
        min_values = x.view(-1,5,self.patch_size*self.patch_size).min(-1)[0].view(-1,self.out_classes,1,1) # Bug, I've been MinMaxScaling with the wrong values
        max_values = x.view(-1,5,self.patch_size*self.patch_size).max(-1)[0].view(-1,self.out_classes,1,1)
        x = (x - min_values)/(max_values - min_values)

        Y = torch.concat([X, x], dim=1).to(X.device)
        y = self.classifier(Y)
        y['masks'] = x

        return y
    


class UNetPreload(nn.Module):
    def __init__(self, in_channels, out_classes, patch_size, encoder_name, classifier_name, classifier_classes, load_dir, fold):
        super(UNetPreload, self).__init__()
        self.patch_size = patch_size
        self.UNet = smp.Unet(
            encoder_name=encoder_name,
            classes=out_classes,
            in_channels=in_channels,
            encoder_weights='imagenet' if rsnautils.PRELOAD else None
        )
        self.encoder_name = encoder_name
        self.out_classes = out_classes

        final_channels = out_classes + in_channels
        self.classifier_classes = classifier_classes
        self.classifier = vision2d.RSNA24Model(model_name=classifier_name, in_c=final_channels, n_classes=classifier_classes)

        self.load(load_dir=load_dir, fold=fold)

    def load(self, load_dir, fold):
        fname = Path(load_dir) / (self.name() + f'_fold{fold}.pth')
        logger.info(f'Loading Model from {fname}')
        self.load_state_dict(torch.load(fname))

        for parameter in self.UNet.parameters():
            parameter.requires_grad = False

    def name(self):
        return f'unet_{self.encoder_name}_{self.classifier.name()}'

    def forward(self,X):
        if isinstance(X, dict):
            key = [k for k in X.keys() if 'Patch' in k][0]
            X = X[key]
        x = self.UNet(X)
#       MinMaxScaling along the class plane to generate a heatmap
        min_values = x.view(-1,5,self.patch_size*self.patch_size).min(-1)[0].view(-1,self.out_classes,1,1) # Bug, I've been MinMaxScaling with the wrong values
        max_values = x.view(-1,5,self.patch_size*self.patch_size).max(-1)[0].view(-1,self.out_classes,1,1)
        x = (x - min_values)/(max_values - min_values)

        Y = torch.concat([X, x], dim=1).to(X.device)
        y = self.classifier(Y)
        y['masks'] = x

        return y
    

class UNetPreloadZoom(nn.Module):
    def __init__(self, in_channels, out_classes, patch_size, encoder_name, classifier_name, classifier_classes, subsize: int, load_dir: str, fold: int):
        super(UNetPreloadZoom, self).__init__()
        self.UNet = UNetPreload(in_channels, out_classes, patch_size, encoder_name, classifier_name, classifier_classes, load_dir, fold)
        
        self.subsize = subsize
        self.classifier = vision2d.RSNA24Model(model_name=classifier_name, in_c=in_channels, n_classes=3)

    def name(self):
        return f'unetzoom_{self.UNet.encoder_name}_{self.classifier.name()}'

    def get_zoom(self, mask, x):
        coord = torch.argwhere(mask > 0.9).float().mean(dim=0).long()
        if self.training:
            coord[0] += np.random.randint(-10, 11)
            coord[1] += np.random.randint(-10, 11)
        X, Y = torch.clip(coord, self.subsize, self.UNet.patch_size - self.subsize)
        return x[:, (X-self.subsize):(X+self.subsize), (Y-self.subsize):(Y+self.subsize)]

    def forward(self,X):
        if isinstance(X, dict):
            key = [k for k in X.keys() if 'Patch' in k][0]
            X = X[key]
        y = self.UNet(X)
        masks = y['masks']
        B = X.shape[0]
        M = masks.shape[1]

        X2 = torch.stack(sum([[self.get_zoom(masks[i, j], X[i]) for j in range(M)] for i in range(B)],[]), dim=0)
        pred2 = self.classifier(X2)

        y['labels'] = pred2['labels'].reshape(B, -1)

        return y