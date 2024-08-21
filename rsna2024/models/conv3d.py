import timm
import logging
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils as rsnautils


from . import unet


logger = logging.getLogger(__name__)

class Residual3D(nn.Module):
    def __init__(self, out_channels, kernel_size=3, use_1x1_conv=False, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.LazyConv3d(out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.conv2 = nn.LazyConv3d(out_channels=out_channels, kernel_size=kernel_size, padding=1)
        if use_1x1_conv:
            self.conv3 = nn.LazyConv3d(out_channels=out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

        self.bn1 = nn.LazyBatchNorm3d()
        self.bn2 = nn.LazyBatchNorm3d()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(num_residuals, num_channels, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual3D(num_channels, use_1x1_conv=True, stride=2))
        else:
            blk.append(Residual3D(num_channels))
    return nn.Sequential(*blk)


class ResNet3D(nn.Module):
    def __init__(self, arch, n_classes):
        super().__init__()
        b1 = nn.Sequential(
            nn.LazyConv3d(64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.LazyBatchNorm3d(), nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1))
        self.net = nn.Sequential(b1)
        for i, b in enumerate(arch):
            self.net.add_module(f'b{i+2}', resnet_block(*b, first_block=(i==0)))
        self.net.add_module('last', nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)), nn.Flatten(), nn.LazyLinear(n_classes)))
    def forward(self, X):
        if len(X.shape) == 4:  # B, D, H, W, good if we are just feeding in a central image.  Otherwise, 
            X = X.unsqueeze(1)
        y = self.net(X)
        return {'labels': y}
            
            
class ResNet3D18(ResNet3D):
    def __init__(self, num_classes=3):
        super().__init__(((4, 64), (2, 128), (2, 256), (2, 512), (2, 1024), (2, 2048)), num_classes)

    def name(self):
        return f'resnet3d_18'