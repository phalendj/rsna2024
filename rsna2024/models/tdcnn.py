import timm
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


from . import unet


logger = logging.getLogger(__name__)


class TDCNNModel(nn.Module):
    def __init__(self, model_name, img_size, in_c=1, n_classes=3, pretrained=True):
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
        Y = self.feature_model.model.forward_features(X)
        d_model = Y.shape[1]
        print(f'Feature dimension {d_model}')
        
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=4)
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
        # x.shape = B, I, H, W, where B == 1, I = # of images
        assert x.shape[0] == 1, 'Batch size must be 1 for TD CNN'
        x = x.transpose(1,0)  # I, 1, H, W
        y = self.feature_model.model.forward_features(x)  # I, 1024, 4 ,4
        y = self.pool(y).flatten(1).unsqueeze(0)  # 1, I, 1024
        y = self.encoder(y)  # 1, I, 1024
        y = F.adaptive_avg_pool1d(y.transpose(-1, -2), 1).squeeze(-1)  # 1, 1024
        return {'labels': self.classifier(y)}
    

class TDCNNUNetPreloadZoom(nn.Module):
    def __init__(self, in_channels, out_classes, patch_size, encoder_name, classifier_name, classifier_classes, subsize: int, load_dir: str, fold: int):
        super(TDCNNUNetPreloadZoom, self).__init__()
        self.UNet = unet.UNetPreload(in_channels, out_classes, patch_size, encoder_name, classifier_name, classifier_classes, load_dir, fold)
        
        self.subsize = subsize
        self.classifier = TDCNNModel(model_name=classifier_name, img_size=(subsize, subsize), in_c=1, n_classes=3)

    def name(self):
        return f'tdcnn_unetzoom_{self.UNet.encoder_name}_{self.classifier.name()}'

    def get_zoom(self, mask, x):
        coord = torch.argwhere(mask > 0.9).float().mean(dim=0).long()
        X, Y = torch.clip(coord, self.subsize, self.UNet.patch_size - self.subsize)
        return x[:, (X-self.subsize):(X+self.subsize), (Y-self.subsize):(Y+self.subsize)]

    def forward(self,X):
        y = self.UNet(X)
        masks = y['masks']
        B = X.shape[0]
        M = masks.shape[1]

        X2 = torch.stack(sum([[self.get_zoom(masks[i, j], X[i]) for j in range(M)] for i in range(B)],[]), dim=0)
        pred2 = self.classifier(X2)

        y['labels'] = pred2['labels'].reshape(B, -1)

        return y