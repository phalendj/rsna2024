import timm
import logging
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .. import utils as rsnautils
except ImportError:
    import utils as rsnautils

from .tdcnn import PositionalEncoding


logger = logging.getLogger(__name__)


class FullLevelTDCNNModel(nn.Module):
    def __init__(self, model_name: str, img_size: tuple[int, int], in_c: int = 1, n_classes: int = 3, num_layers: int = 4):
        super().__init__()
        self.feature_model = timm.create_model(
                                    model_name,
                                    pretrained=rsnautils.PRELOAD, 
                                    features_only=False,
                                    in_chans=in_c,
                                    num_classes=n_classes,
                                    global_pool='avg'
                                    )
        X = torch.randn(2, 1, *img_size)
        Y = self.feature_model.forward_features(X)

        asize = 1

        self.d_model = Y.shape[1] * asize**2
        logger.info(f'Feature dimension for tdcnn {self.d_model}')
        
        self.pool = nn.AdaptiveMaxPool2d(output_size=asize)
        self.pos_encoding = PositionalEncoding(num_hiddens=self.d_model, dropout=0.0)
        layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.LazyLinear(n_classes)
        self.nclasses = n_classes

        #TODO: Add dropouts

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
        return f'full_level_td_cnn_{self.model_name}'

    def load(self, load_dir, fold):
        fname = Path(load_dir) / (self.name() + f'_fold{fold}.pth')
        logger.info(f'Loading Model from {fname}')
        self.load_state_dict(torch.load(fname, weights_only=True))
        self.classifier.reset_parameters()

    def forward_features(self, x):
        B, L, I, H, W = x.shape
        # print(x.reshape(-1, H, W).unsqueeze(1).shape)
        # x = x.unsqueeze(3).flatten(0,2)
        # print(x.shape)
        x = x.flatten(0,2).unsqueeze(1)
        y = self.feature_model.forward_features(x)  # B*L*I, D, 4 ,4
        y = self.pool(y).flatten(1)  # B*L*I, D  # TODO: Could make this part bigger?
        y = y.reshape(B, L, I, self.d_model)
        return y

    def forward(self, x):
        # return self.tdcnn(x['Sagittal T2/STIR Patch'])
        sagittal_t2_mask = (x['Sagittal T2/STIR Instance Numbers'] != -1)  # B, L, I
        sagittal_t1_mask = (x['Sagittal T1 Instance Numbers'] != -1)
        axial_t2_mask = (x['Axial T2 Instance Numbers'] != -1)

        __, __, I_sag_t2 = sagittal_t2_mask.shape
        __, __, I_sag_t1 = sagittal_t1_mask.shape
        __, __, I_ax_t2 = axial_t2_mask.shape

        sagittal_t2_features = self.forward_features(x['Sagittal T2/STIR Patch'])  # B, L, I, D        
        sagittal_t1_features = self.forward_features(x['Sagittal T1 Patch'])
        axial_t2_features = self.forward_features(x['Axial T2 Patch'])

        # TODO:
        # Add src mask as extra part of the vector?
        # Add instance label prediction as part of feature vector?

        y = torch.concat([sagittal_t2_features, sagittal_t1_features, axial_t2_features], dim=2)  # B, L, I1 + I2 +I3, D
        # mask = torch.concat([sagittal_t2_mask, sagittal_t1_mask, axial_t2_mask], dim=2)    # B, L, I1 + I2 +I3
        # y = sagittal_t2_features
        B, L, I, D = y.shape
        assert D == self.d_model
        y = y.flatten(0, 1)
        y = self.pos_encoding(y)  # B*L, ISum, D
        # y = self.encoder(y, src_key_padding_mask=mask.flatten(0, 1)) # B*L, ISum, D
        y = self.encoder(y) # B*L, ISum, D
        # y_sag_t2 = F.adaptive_avg_pool1d(y.transpose(-1, -2)[..., :I_sag_t2], 1).squeeze(-1)  # B*L, D
        # y_sag_t1 = F.adaptive_avg_pool1d(y.transpose(-1, -2)[..., I_sag_t2:-I_ax_t2], 1).squeeze(-1)  # B*L, D
        # y_ax_t2 = F.adaptive_avg_pool1d(y.transpose(-1, -2)[..., -I_ax_t2:], 1).squeeze(-1)  # B*L, D
        # y = self.classifier(torch.concat([y_sag_t2, y_sag_t1, y_ax_t2], dim=1))  # B*L, n_classes
        y = F.adaptive_max_pool1d(y.transpose(-1,-2), 1).flatten(1)  # TODO: Make this pooling larger?
        y = self.classifier(y)
        y = y.reshape(B, L, self.nclasses // 3, 3)  # Now B, Level, Condition, diagnosis
        y = y.transpose(1,2).flatten(1)  # Now B, nclasses*nlevels

        return {'labels': y}

