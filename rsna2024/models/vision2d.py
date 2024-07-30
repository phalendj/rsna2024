import timm
import torch.nn as nn


class RSNA24Model(nn.Module):
    def __init__(self, model_name, in_c=30, n_classes=75, pretrained=True, features_only=False):
        super().__init__()
        self.model = timm.create_model(
                                    model_name,
                                    pretrained=pretrained, 
                                    features_only=features_only,
                                    in_chans=in_c,
                                    num_classes=n_classes,
                                    global_pool='avg'
                                    )
        self.model_name = model_name
    
    def name(self):
        return f'vision2d_rsna2024_{self.model_name}'

    def forward(self, x):
        y = self.model(x)
        return y