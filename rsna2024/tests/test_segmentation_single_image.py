import torch
import numpy as np

from torch.utils.data import DataLoader

import datasets.segmentation_single_image as segmentationsingle
import datasets.augmentations as aug
import utils as rsnautils


def test_dataset():
    rsnautils.set_clean(False)
    ds = segmentationsingle.SegmentationCenterDataset(study_ids=[4003253,    4646740,    7143189],
                                                                 image_size=[512, 512],
                                                                 channels=1,
                                                                 conditions=['Spinal Canal Stenosis'],
                                                                 series_description='Sagittal T2/STIR',
                                                                 mode='train', 
                                                                 aug_size=0.1,
                                                                 transform=None
                                                            )
    
    data, t = ds[0]
    x = data['Sagittal T2/STIR Patch']

    assert x.shape == torch.Size([1, 512, 512])
    assert t['centers'].shape == torch.Size([5, 2])
    assert t['labels'].shape == torch.Size([5])
    assert t['study_id'] == torch.tensor([4003253])


def test_dataset2():
    rsnautils.set_clean(False)
    ds = segmentationsingle.SegmentationCenterDataset(study_ids=[4003253,    4646740,    7143189],
                                                                 image_size=[512, 512],
                                                                 channels=3,
                                                                 conditions=['Spinal Canal Stenosis'],
                                                                 series_description='Sagittal T2/STIR',
                                                                 mode='train', 
                                                                 aug_size=0.1,
                                                                 transform=None
                                                            )
    
    data, t = ds[0]

    x = data['Sagittal T2/STIR Patch']

    assert x.shape == torch.Size([3, 512, 512])
    assert t['centers'].shape == torch.Size([5, 2])
    assert t['labels'].shape == torch.Size([5])
    assert t['study_id'] == torch.tensor([4003253])


def test_dataset3():
    rsnautils.set_clean(False)
    ds = segmentationsingle.SegmentationCenterDataset(study_ids=[4003253,    4646740,    7143189],
                                                                 image_size=[512, 512],
                                                                 channels=5,
                                                                 conditions=['Spinal Canal Stenosis'],
                                                                 series_description='Sagittal T2/STIR',
                                                                 mode='train', 
                                                                 aug_size=0.1,
                                                                 transform=None
                                                            )
    
    data, t = ds[0]
    x = data['Sagittal T2/STIR Patch']
    
    assert x.shape == torch.Size([5, 512, 512])
    assert t['centers'].shape == torch.Size([5, 2])
    assert t['slice_classification'].shape == torch.Size([5])
    assert np.all(t['slice_classification'].numpy() == np.array([0,0,1,0,0]))
    assert t['labels'].shape == torch.Size([5])
    assert t['study_id'] == torch.tensor([4003253])


def test_dataset4():
    rsnautils.set_clean(False)
    ds = segmentationsingle.SegmentationPredictedCenterDataset(study_ids=[4003253,    4646740,    7143189],
                                                                 image_size=[512, 512],
                                                                 channels=5,
                                                                 conditions=['Spinal Canal Stenosis'],
                                                                 series_description='Sagittal T2/STIR',
                                                                 generated_coordinate_file='/data/phalendj/kaggle/rsna2024/train_label_coordinates.csv',
                                                                 mode='train', 
                                                                 aug_size=0.1,
                                                                 transform=None
                                                            )
    
    data, t = ds[0]
    x = data['Sagittal T2/STIR Patch']
    assert x.shape == torch.Size([5, 512, 512])
    assert t['centers'].shape == torch.Size([5, 2])
    assert t['slice_classification'].shape == torch.Size([5])
    assert np.all(t['slice_classification'].numpy() == np.array([0,0,1,0,0]))
    assert t['labels'].shape == torch.Size([5])
    assert t['study_id'] == torch.tensor([4003253])


def test_dataloader():
    ds = segmentationsingle.SegmentationCenterDataset(study_ids=[4003253,    4646740,    7143189],
                                                                 image_size=[512, 512],
                                                                 channels=1,
                                                                 conditions=['Spinal Canal Stenosis'],
                                                                 series_description='Sagittal T2/STIR',
                                                                 mode='train', 
                                                                 aug_size=0.1,
                                                                 transform=None
                                                            )
    
    dl = DataLoader(ds, batch_size=3, shuffle=False, pin_memory=True, drop_last=True, num_workers=1)
    
    data, t = next(iter(dl))

    x = data['Sagittal T2/STIR Patch']

    assert x.shape == torch.Size([3, 1, 512, 512])
    assert t['centers'].shape == torch.Size([3, 5, 2])
    assert t['labels'].shape == torch.Size([3, 5])
    assert torch.all(t['study_id'] == torch.tensor([[4003253], [4646740], [7143189]]))
