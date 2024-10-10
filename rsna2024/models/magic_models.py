## From https://www.kaggle.com/code/hengck23/ver-1-magic-single-stage-model/notebook
## From https://www.kaggle.com/code/hengck23/ver-2-more-magic-single-stage-model/notebook

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import logging

import timm
try:
    from .. import utils as rsnautils
except ImportError:
    import utils as rsnautils

from .tdcnn import PositionalEncoding


logger = logging.getLogger(__name__)



class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)



class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        scaling=2,
        use_batchnorm=True,
    ):
        self.scaling = scaling
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = nn.Identity(in_channels=in_channels + skip_channels)
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = nn.Identity(in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=self.scaling, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x
    
class FlattenerUnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        skip_channels,
        decoder_channels,
        scalings
    ):
        super().__init__()
        # [print(in_ch, skip_ch, out_ch, scl) for in_ch, skip_ch, out_ch, scl in zip(encoder_channels, skip_channels, decoder_channels, scalings)]
        blocks = [DecoderBlock(in_ch, skip_ch, out_ch, scl) for in_ch, skip_ch, out_ch, scl in zip(encoder_channels, skip_channels, decoder_channels, scalings)]
        self.blocks = nn.ModuleList(blocks)
        
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)

    def flattener(self, x):
        x = x.permute(0,2, 3, 4, 1)
        B, D, H, W, I = x.shape
        x = self.pool(x.flatten(1,3))
        x = x.reshape(B, D, H, W)
        return x
        
    def forward(self, x, skips):
        skips = skips[::-1]  # reverse feature pyramid on skips

        x = self.flattener(x)
        for i, decoder_block in enumerate(self.blocks):
            skip = self.flattener(skips[i]) if i < len(skips) else None
            x = decoder_block(x, skip)

        return x


class InstancePredictionHead(nn.Module):
    def __init__(self, d_model, num_layers, dropout, output_size, n_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=output_size)
        self.pos_encoding = PositionalEncoding(num_hiddens=d_model, dropout=dropout)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.LazyLinear(n_classes)
        
    def forward(self, X):
        B, I, D, H, W = X.shape
        y = self.pool(X.flatten(0,1)).flatten(1)
        y = y.reshape(B, I, -1)
        y = self.pos_encoding(y)
        y = self.encoder(y)  # B, I, 1024
        y = self.dropout(y)
        return self.classifier(y)



class MagicModel(nn.Module):
    def __init__(self, bottleneck_layers, instance_layers, model_name, dropout, num_points, n_classes):
        super().__init__()

        self.feature_model = timm.create_model(
                                                model_name,
                                                pretrained=rsnautils.PRELOAD, 
                                                features_only=True,
                                                in_chans=1,
                                                num_classes=0,
                                                global_pool=''
                                                )
        
        self.model_name = model_name
        img_size = (512, 512)
        X = torch.randn(2, 1, *img_size)
        X = self.feature_model(X)
        
        self.encoder_dim = [x.shape[1] for x in X]
        logger.info(f'Encoder dimension for {model_name} {self.encoder_dim}')

        # encoder_channels = list(reversed(self.encoder_dim[:-1]))
        encoder_channels =  {'densenet121': [1024,512,256,64],
                             'pvt_v2_b2': [512,320,128,64],
                             'tf_efficientnetv2_b2.in1k': [208, 120, 56, 32, 16]}[model_name]
        # encoder_channels.append(encoder_channels[-1])

        skip_channels =  {'densenet121': [1024,512,256,64],
                          'pvt_v2_b2': [320,128,64, 0],
                          'tf_efficientnetv2_b2.in1k': [120, 56, 32, 16, 0]}[model_name]
        
        scalings =  {'densenet121': [2,2,2,2,2],
                     'pvt_v2_b2': [2,2,2,4],
                     'tf_efficientnetv2_b2.in1k': [2,2,2,2,2]}[model_name]
                     

        decoder_channels = {'densenet121': [512, 256, 64, 64],
                            'pvt_v2_b2': [320, 128, 64, 64],
                            'tf_efficientnetv2_b2.in1k': [120, 56, 32, 16]
                            }[model_name]

        while len(decoder_channels) < len(encoder_channels):
            decoder_channels.append(decoder_channels[-1])

        logger.info(f'Decoder channels {decoder_channels}, {encoder_channels}')

        d_model = self.encoder_dim[-1]
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(layer, num_layers=bottleneck_layers)

        output_size = int(np.sqrt(1024 // d_model))
        logger.info(f'Adaptive pooling size: {output_size} for d_model = {d_model}')

        self.instance_prediction = InstancePredictionHead(d_model=d_model*output_size, num_layers=instance_layers, dropout=dropout, output_size=output_size, n_classes=num_points*2)

        self.unet_decoder = FlattenerUnetDecoder(encoder_channels=encoder_channels, skip_channels=skip_channels, decoder_channels=decoder_channels, scalings=scalings)

        self.num_points = num_points
        final_channels = decoder_channels[-1]
        self.heatmap_mask = nn.Conv2d(in_channels=final_channels, kernel_size=1, out_channels=num_points)
        self.grade_mask = nn.Conv2d(in_channels=final_channels, kernel_size=1, out_channels=final_channels)

        self.grader = nn.Linear(final_channels, n_classes)
        # self.grader = nn.Sequential(
        #     nn.Linear(final_channels, final_channels),
        #     nn.BatchNorm1d(num_points),
        #     nn.ReLU(),
        #     nn.Linear(final_channels, n_classes),
        # )
        
    def reset_grader(self):
        pass
        # self.grader[0].reset_parameters()
        # self.grader[-1].reset_parameters()
   
    def name(self):
        return f'magic_{self.model_name}'

    def forward(self, X):
        if isinstance(X, dict):
            key = [k for k in X.keys() if 'Patch' in k][0]
            X = X[key]


        B, I, H, W = X.shape
        X = X.flatten(0, 1)
        feature_maps = self.feature_model(X.unsqueeze(1))
        x = feature_maps[-1]
        Bp, Cp, Hp, Wp = x.shape
        x = x.permute(0, 2, 3, 1).flatten(1,2)
        x = self.encoder(x)
        x = x.reshape(B*I, Hp, Wp, Cp).permute(0, 3, 1, 2).reshape(B, I, Cp, Hp, Wp)

        skips = [s.reshape(B, I, *s.shape[1:]) for s in feature_maps[:-1]]

        xt = self.unet_decoder(x, skips)
        heatmap = self.heatmap_mask(xt)
        grades = self.grade_mask(xt)

        # Somehow this fails in mixed precision  It could be an overflow in float16 type
        to_grade = (heatmap.unsqueeze(2)*grades.unsqueeze(1)).mean(dim=(3,4)) # .type(heatmap.dtype)
        if np.isnan(to_grade.mean().item()):
            print('Grade Fail', to_grade.shape, to_grade.mean())
        pred = self.grader(to_grade).flatten(1)
        if np.isnan(pred.mean().item()):
            print('Pred Fail', pred.shape, pred.mean())
        instance_predictions = self.instance_prediction(x)

        min_values = heatmap.view(-1,self.num_points,H*W).min(-1)[0].view(-1,self.num_points,1,1) # Bug, I've been MinMaxScaling with the wrong values
        max_values = heatmap.view(-1,self.num_points,H*W).max(-1)[0].view(-1,self.num_points,1,1)
        heatmap = (heatmap - min_values)/(max_values - min_values)

        return {'instance_labels': instance_predictions, 'labels': pred, 'masks': heatmap}



