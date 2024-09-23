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


logger = logging.getLogger(__name__)


class LSTMCNNModel(nn.Module):
    def __init__(self, model_name: str, img_size: tuple[int, int], in_c: int = 1, n_classes: int = 3, num_layers: int = 4, dropout: float = 0.1):
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
        d_model = Y.shape[1]
        logger.info(f'Feature dimension for tdcnn {d_model}')
        
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.encoder = nn.LSTM(d_model, 512, num_layers=num_layers, dropout=dropout, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
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
        return f'lstm_cnn_{self.model_name}'
    
    def forward_encode(self, x):
        B, I, H, W = x.shape
        x = x.unsqueeze(2).flatten(0,1)
        y = self.feature_model.forward_features(x)  # B*I, D, 4 ,4
        y = self.pool(y).flatten(1)  # B*I, D
        y = y.reshape(B, I, -1)
        y, __ = self.encoder(y)  # B, I, 1024
        return y

    def forward_features(self, x):
        # B, I, H, W = x.shape
        y = self.forward_encode(x)  # B, I, 1024
    
        #TODO: Try different pooling methods: avg, max, catavgmax
        y = F.adaptive_avg_pool1d(y.transpose(-1, -2), 1).squeeze(-1)  # B, 1024
        # y = F.adaptive_max_pool1d(y.transpose(-1, -2), 1).squeeze(-1)  # B, 1024
        # y = torch.concatenate([F.adaptive_avg_pool1d(y.transpose(-1, -2), 1).squeeze(-1), F.adaptive_max_pool1d(y.transpose(-1, -2), 1).squeeze(-1)], dim=1)
        return y

    def forward(self, X):
        if isinstance(X, dict):
            key = [k for k in X.keys() if 'Patch' in k][0]
            X = X[key]
        y = self.forward_features(X)
        y = self.dropout(y)
        return {'labels': self.classifier(y)}


class LSTMCNNInstanceModel(LSTMCNNModel):
    def name(self):
        return f'lstm_cnn_instance_{self.model_name}'
    
    def forward(self, X):
        if isinstance(X, dict):
            key = [k for k in X.keys() if 'Patch' in k][0]
            X = X[key]
        y = self.forward_encode(X)
        y = self.dropout(y)
        return {'instance_labels': self.classifier(y)}



class LSTMCNNLevelModel(LSTMCNNModel):
    def level_forward(self, x):
        B, L, I, H, W = x.shape
        x = x.flatten(0, 1)  # Now first index is is B0L0, B0L1, ..., B0L4, B1L0, B1L1, ... , dim = (B*L, I, H, W)
        t = super().forward(x)
        y = t['labels']
        # y.shape = B*L, nclasses
        y = y.reshape(B, L, -1, 3)  # Now B, Level, Condition, diagnosis
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
        return f'lstm_cnn_level_{self.model_name}'

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

    def forward(self, X):
        if isinstance(X, dict):
            key = [k for k in X.keys() if 'Patch' in k][0]
            X = X[key]
        y = self.level_forward(X)
        return {'labels': y}
    

class LSTMCNNLevelSideModel(LSTMCNNModel):
    def level_forward(self, x):
        B, L, S, I, H, W = x.shape
        assert S == 2
        x = x.flatten(0, 2)  # Now first index is is B0L0S0, B0L0S1, B0L1S0, ..., B0L4S1, B1L0S0, B1L0S1,B1L1S0, ... , dim = (B*L*S, I, H, W)
        t = super().forward(x)
        y = t['labels']
        # y.shape = B*L*S, nclasses
        y = y.reshape(B, L, S, -1)  # Now B, Level, Side, diagnosis
        assert y.shape[-1] == 3
        y = y.transpose(1,2).flatten(1)  # Now B, nclasses*nlevels
        return y
    
    def level_forward_features(self, x):
        B, L, S, I, H, W = x.shape
        x = x.flatten(0, 2)  # Now first index is is B0L0, B0L1, ..., B0L4, B1L0, B1L1, ... , dim = (B*L, I, H, W)
        y = super().forward_features(x)
        # y.shape = B*L, feature dim (1024 or something like that)
        y = y.reshape(B, L, S, -1)  # Now B, Level, feature dim
        return y

    def name(self):
        return f'lstm_cnn_level_side_{self.model_name}'

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

    def forward(self, X):
        if isinstance(X, dict):
            key = [k for k in X.keys() if 'Patch' in k][0]
            X = X[key]
        y = self.level_forward(X)
        return {'labels': y}

