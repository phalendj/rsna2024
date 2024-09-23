## From https://www.kaggle.com/code/hengck23/ver-1-magic-single-stage-model/notebook
## From https://www.kaggle.com/code/hengck23/ver-2-more-magic-single-stage-model/notebook

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

import timm


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
        use_batchnorm=True,
        attention_type=None,
    ):
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
        x = F.interpolate(x, scale_factor=2, mode="nearest")
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
        decoder_channels,
    ):
        super().__init__()
        
        skip_channels = encoder_channels[:-1] + [0]
        blocks = [DecoderBlock(in_ch, skip_ch, out_ch) for in_ch, skip_ch, out_ch in zip(encoder_channels, skip_channels, decoder_channels)]
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
    def __init__(self, d_model, num_layers, dropout, n_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
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

        encoder_channels = list(reversed(self.encoder_dim[:-1]))
        encoder_channels.append(encoder_channels[-1])

        decoder_channels = []
        for i in range(1, len(encoder_channels)):
            if len(decoder_channels) == 0 and encoder_channels[i] == encoder_channels[i-1]:
                pass
            else:
                decoder_channels.append(encoder_channels[i])

        while len(decoder_channels) < len(encoder_channels):
            decoder_channels.append(decoder_channels[-1])

        logger.info(f'Decoder channels {decoder_channels}')

        d_model = self.encoder_dim[-1]
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(layer, num_layers=bottleneck_layers)

        self.instance_prediction = InstancePredictionHead(d_model=d_model, num_layers=instance_layers, dropout=dropout, n_classes=num_points*2)

        self.unet_decoder = FlattenerUnetDecoder(encoder_channels=encoder_channels, decoder_channels=decoder_channels)

        final_channels = decoder_channels[-1]
        self.heatmap_mask = nn.Conv2d(in_channels=final_channels, kernel_size=1, out_channels=num_points)
        self.grade_mask = nn.Conv2d(in_channels=final_channels, kernel_size=1, out_channels=final_channels)

        self.grader = nn.Sequential(
            nn.Linear(final_channels, final_channels),
            nn.BatchNorm1d(num_points),
            nn.ReLU(inplace=True),
            nn.Linear(final_channels, n_classes),
        )

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

        to_grade = (heatmap.unsqueeze(2)*grades.unsqueeze(1)).sum(dim=(3,4))
        pred = self.grader(to_grade)

        instance_predictions = self.instance_prediction(x)

        return {'instance_labels': instance_predictions, 'labels': pred, 'masks': heatmap}



