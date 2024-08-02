import torch
import pytest

from torch.utils.data import DataLoader

import models.tdcnn as tdcnn


def test_output():
    X = torch.randn((3, 15, 512, 512))
    model = tdcnn.TDCNNUNetPreloadZoom(in_channels=5, out_classes=5, patch_size=512, encoder_name='resnet34', classifier_name='densenet201', classifier_classes=15, subsize=64, 
                                       load_dir='/home/phalendj/code/rsna2024/outputs/2024-08-01/15-48-48/', fold=0, predict_classes=3)
    model.eval()
    y = model(X)
    assert y['masks'].shape == torch.Size([3, 5, 512, 512])
    assert y['labels'].shape == torch.Size([3, 15])
    


def test_output_double():
    X = torch.randn((3, 20, 512, 512))
    model = tdcnn.TDCNNUNetPreloadZoom(in_channels=7, out_classes=10, patch_size=512, encoder_name='resnet34', classifier_name='densenet201', classifier_classes=30, subsize=64, 
                                       load_dir='/home/phalendj/code/rsna2024/outputs/2024-08-02/12-57-24/', fold=0, predict_classes=6)
    model.eval()
    y = model(X)
    assert y['masks'].shape == torch.Size([3, 10, 512, 512])
    assert y['labels'].shape == torch.Size([3, 60])
    