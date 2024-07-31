import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from . import vision2d


class UNet(nn.Module):
    def __init__(self, in_channels, out_classes, patch_size, encoder_name, classifier_name, classifier_classes):
        super(UNet, self).__init__()
        self.patch_size = patch_size
        self.UNet = smp.Unet(
            encoder_name=encoder_name,
            classes=out_classes,
            in_channels=in_channels
        )
        self.encoder_name = encoder_name

        final_channels = out_classes * in_channels
        self.classifier_classes = classifier_classes
        self.classifier = vision2d.RSNA24Model(model_name=classifier_name, in_c=final_channels, n_classes=classifier_classes)

    def name(self):
        return f'unet_{self.encoder_name}_{self.classifier.name()}'

    def forward(self,X):
        x = self.UNet(X)
#       MinMaxScaling along the class plane to generate a heatmap
        min_values = x.view(-1,5,self.patch_size*self.patch_size).min(-1)[0].view(-1,5,1,1) # Bug, I've been MinMaxScaling with the wrong values
        max_values = x.view(-1,5,self.patch_size*self.patch_size).max(-1)[0].view(-1,5,1,1)
        x = (x - min_values)/(max_values - min_values)

        Y = torch.concat([X + x[:, i].unsqueeze(1) for i in range(x.shape[1])], dim=1).to(X.device)
        y = self.classifier(Y)

        return {'masks': x, 'labels': y}