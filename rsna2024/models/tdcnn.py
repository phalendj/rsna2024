import timm
import logging
import numpy as np
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

    def forward(self, x):
        # x.shape = B, I, H, W, 
        B, I, H, W = x.shape
        x = x.unsqueeze(2).flatten(0,1)
        y = self.feature_model.forward_features(x)  # B*I, D, 4 ,4
        y = self.pool(y).flatten(1)  # B*I, D
        y = y.reshape(B, I, -1)
        y = self.encoder(y)  # B, I, 1024
        y = F.adaptive_avg_pool1d(y.transpose(-1, -2), 1).squeeze(-1)  # B, 1024
        return {'labels': self.classifier(y)}
    

class TDCNNUNetPreloadZoom(nn.Module):
    def __init__(self, in_channels, out_classes, patch_size, encoder_name, classifier_name, classifier_classes, subsize: int, load_dir: str, fold: int):
        super(TDCNNUNetPreloadZoom, self).__init__()
        self.unet_in_channels = in_channels
        self.UNet = unet.UNetPreload(in_channels, out_classes, patch_size, encoder_name, classifier_name, classifier_classes, load_dir, fold)
        
        self.subsize = subsize
        self.classifier = TDCNNModel(model_name=classifier_name, img_size=(subsize, subsize), in_c=1, n_classes=3)

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