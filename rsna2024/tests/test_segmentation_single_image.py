import torch
import pytest

from torch.utils.data import DataLoader

import datasets.segmentation_single_image as segmentationsingle
import datasets.augmentations as aug

def test_dataset():
    ds = segmentationsingle.SegmentationSingleImageDataset(study_ids=[4003253,    4646740,    7143189],
                                                                 image_size=[512, 512],
                                                                 conditions=['Spinal Canal Stenosis'],
                                                                 series_description='Sagittal T2/STIR',
                                                                 mode='train', 
                                                                 aug_size=0.1
                                                            )
    
    x, t = ds[0]
    assert x.shape == torch.Size([1, 512, 512])
    assert t['centers'].shape == torch.Size([5, 2])
    assert t['labels'].shape == torch.Size([5])
    assert t['study_id'] == torch.tensor([4003253])


def test_dataset2():
    ds = segmentationsingle.SegmentationCenterDataset(study_ids=[4003253,    4646740,    7143189],
                                                                 image_size=[512, 512],
                                                                 channels=3,
                                                                 conditions=['Spinal Canal Stenosis'],
                                                                 series_description='Sagittal T2/STIR',
                                                                 mode='train', 
                                                                 aug_size=0.1,
                                                                 transform=None
                                                            )
    
    x, t = ds[0]
    assert x.shape == torch.Size([3, 512, 512])
    assert t['centers'].shape == torch.Size([5, 2])
    assert t['labels'].shape == torch.Size([5])
    assert t['study_id'] == torch.tensor([4003253])


def test_dataset3():
    ds = segmentationsingle.SegmentationCenterDataset(study_ids=[4003253,    4646740,    7143189],
                                                                 image_size=[512, 512],
                                                                 channels=5,
                                                                 conditions=['Spinal Canal Stenosis'],
                                                                 series_description='Sagittal T2/STIR',
                                                                 mode='train', 
                                                                 aug_size=0.1,
                                                                 transform=None
                                                            )
    
    x, t = ds[0]
    assert x.shape == torch.Size([5, 512, 512])
    assert t['centers'].shape == torch.Size([5, 2])
    assert t['labels'].shape == torch.Size([5])
    assert t['study_id'] == torch.tensor([4003253])


def test_dataloader():
    ds = segmentationsingle.SegmentationSingleImageDataset(study_ids=[4003253,    4646740,    7143189],
                                                                 image_size=[512, 512],
                                                                 conditions=['Spinal Canal Stenosis'],
                                                                 series_description='Sagittal T2/STIR',
                                                                 mode='train', 
                                                                 aug_size=0.1
                                                            )
    
    dl = DataLoader(ds, batch_size=3, shuffle=False, pin_memory=True, drop_last=True, num_workers=1)
    
    x, t = next(iter(dl))

    assert x.shape == torch.Size([3, 1, 512, 512])
    assert t['centers'].shape == torch.Size([3, 5, 2])
    assert t['labels'].shape == torch.Size([3, 5])
    assert torch.all(t['study_id'] == torch.tensor([[4003253], [4646740], [7143189]]))
