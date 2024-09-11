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

class Conv2_1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1),
                        padding=(0, 0, 0), dilation=(1, 1, 1), bias=True):
        super().__init__()
        t = kernel_size[0]
        d = (kernel_size[1] + kernel_size[2])//2
        self.in_channels = in_channels
        self.out_channels = out_channels

        #Hidden size estimation to get a number of parameter similar to the 3d case
        self.hidden_size = int((t*d**2*in_channels*out_channels)/(d**2*in_channels+t*out_channels))

        self.conv2d = nn.Conv2d(in_channels, self.hidden_size, kernel_size[1:], stride[1:], padding[1:], bias=bias)
        self.conv1d = nn.Conv1d(self.hidden_size, out_channels, kernel_size[0], stride[0], padding[0], bias=bias)

    def forward(self, x):
        #2D convolution
        b, c, t, d1, d2 = x.size()
        x = x.permute(0,2,1,3,4).contiguous()
        x = x.view(b*t, c, d1, d2)
        x = F.relu(self.conv2d(x))
        
        #1D convolution
        c, dr1, dr2 = x.size(1), x.size(2), x.size(3)
        x = x.view(b, t, c, dr1, dr2)
        x = x.permute(0, 3, 4, 2, 1).contiguous()
        x = x.view(b*dr1*dr2, c, t)
        x = self.conv1d(x)

        #Final output
        out_c, out_t = x.size(1), x.size(2)
        x = x.view(b, dr1, dr2, out_c, out_t)
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        return x

class ConvTranspose2_1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1),
                        padding=(0, 0, 0), dilation=(1, 1, 1), bias=True):
        super().__init__()
        t = kernel_size[0]
        d = (kernel_size[1] + kernel_size[2])//2
        self.in_channels = in_channels
        self.out_channels = out_channels

        #Hidden size estimation to get a number of parameter similar to the 3d case
        self.hidden_size = int((t*d**2*in_channels*out_channels)/(d**2*in_channels+t*out_channels))

        self.convTranspose2d = nn.ConvTranspose2d(in_channels, self.hidden_size, kernel_size[1:], stride[1:], padding[1:], bias=bias)
        self.convTranspose1d = nn.ConvTranspose1d(self.hidden_size, out_channels, kernel_size[0], stride[0], padding[0], bias=bias)

    def forward(self, x):
        #2D convolution
        b, c, t, d1, d2 = x.size()
        x = x.permute(0,2,1,3,4).contiguous()
        x = x.view(b*t, c, d1, d2)
        x = F.relu(self.convTranspose2d(x))
        
        #1D convolution
        c, dr1, dr2 = x.size(1), x.size(2), x.size(3)
        x = x.view(b, t, c, dr1, dr2)
        x = x.permute(0, 3, 4, 2, 1).contiguous()
        x = x.view(b*dr1*dr2, c, t)
        x = self.convTranspose1d(x)

        #Final output
        out_c, out_t = x.size(1), x.size(2)
        x = x.view(b, dr1, dr2, out_c, out_t)
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        return x




class Residual2p1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3,3), use_1x1_conv=False, stride=(1,1,1), padding=(1,1,1)):
        super().__init__()
        self.conv1 = Conv2_1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=(1,1,1))
        self.conv2 = Conv2_1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=(1,1,1), padding=padding)
        if use_1x1_conv:
            self.conv3 = Conv2_1d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1,1), stride=stride)
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


def resnet_block(num_residuals, in_channels, num_channels, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual2p1D(in_channels=in_channels, out_channels=num_channels, use_1x1_conv=True, stride=(1,2,2)))
        else:
            blk.append(Residual2p1D(in_channels=num_channels, out_channels=num_channels))
    return nn.Sequential(*blk)


class ResNet2p1D(nn.Module):
    def __init__(self, arch, n_classes):
        super().__init__()
        b1 = nn.Sequential(
            Conv2_1d(in_channels=1, out_channels=64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
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
            
        elif len(X.shape) == 5 and X.shape[1] == 5:  # B, L, D, H, W is a level model, reshape appropriately
            B, L, I, H, W = X.shape
            X = X.flatten(0, 1)  # Now first index is is B0L0, B0L1, ..., B0L4, B1L0, B1L1, ... , dim = (B*L, I, H, W)
            X = X.unsqueeze(1)
            y = self.net(X)
            y = y.reshape(B, L, -1)  # Now B, Level, feature dim
            y = y.transpose(1,2).flatten(1)  # Now B, nclasses*nlevels
        elif len(X.shape) == 6 and X.shape[1] == 5 and X.shape[2] == 2:  # B, L, S, D, H, W is a level side model, reshape appropriately
            B, L, S, I, H, W = X.shape
            X = X.flatten(0, 2)  # Now first index is is B0L0S0, B0L0S1, B0L1S0, ..., B0L4S1, B1L0S0, B1L0S1,B1L1S0, ... , dim = (B*L*S, I, H, W)
            X = X.unsqueeze(1)
            y = self.net(X)
            y = y.reshape(B, L, S, -1)  # Now B, Level, feature dim
            y = y.transpose(1,2).flatten(1)  # Now B, nclasses*nlevels

        return {'labels': y}
            
            
class ResNet2p1D18(ResNet2p1D):
    def __init__(self, num_classes=3):
        super().__init__(arch=((4, 64, 64), (2, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 1024), (2, 1024, 2048)), n_classes=num_classes)

    def name(self):
        return f'resnet2p1d_18'