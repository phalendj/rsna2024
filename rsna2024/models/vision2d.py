import timm
import torch.nn as nn
import utils as rsnautils


class RSNA24Model(nn.Module):
    def __init__(self, model_name, in_c=30, n_classes=75, features_only=False):
        super().__init__()
        self.model = timm.create_model(
                                    model_name,
                                    pretrained=rsnautils.PRELOAD, 
                                    features_only=features_only,
                                    in_chans=in_c,
                                    num_classes=n_classes,
                                    global_pool='avg'
                                    )
        self.model_name = model_name
    
    def name(self):
        return f'vision2d_rsna2024_{self.model_name}'

    def forward(self, x):
        if len(x.shape) == 4:
            y = self.model(x)
            return {'labels': y}
        else:  # In case of level set data
            B, L, C, H, W = x.shape
            x = x.flatten(0, 1)  # Now first index is is B0L0, B0L1, ..., B0L4, B1L0, B1L1, ... , dim = (B*L, I, H, W)
            y = self.model(x)
            y = y.reshape(B, L, -1, 3)  # Now B, Condition, Level, diagnosis
            y = y.transpose(1,2).flatten(1)  # Now B, nclasses*nlevels
            return {'labels': y}
