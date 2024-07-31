import torch
import pytest

from torch.utils.data import DataLoader

import models.unet as unet


def test_output():
    X = torch.randn((3, 1, 512, 512))
    model = unet.UNet(in_channels=1, out_classes=5, patch_size=512, encoder_name='resnet18', classifier_name='densenet121', classifier_classes=15)
    model.eval()
    y = model(X)
    assert y['masks'].shape == torch.Size([3, 5, 512, 512])
    assert y['labels'].shape == torch.Size([3, 15])
    