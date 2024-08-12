import torch
import pytest

from torch.utils.data import DataLoader

import datasets.level_cubes as level_cubes
import datasets.augmentations as aug


class CROP:
    use = False
    size = 512

class CFG:
    hflip = False
    contrast = True
    blur = True
    distort = True
    rotate = True
    normalize=True
    prob = 0.75
    sharpen = False
    channel_shuffle = False
    crop = CROP


def test_dataset_sagittalt2():
    CH = 7
    SS = 64
    ds = level_cubes.LevelCubeDataset(study_ids=[4003253,    4646740,    7143189],
                                      channels=CH,
                                      slice_size=SS,
                                      conditions=['Spinal Canal Stenosis',     
                                                  'Left Neural Foraminal Narrowing', 
                                                  'Right Neural Foraminal Narrowing',
                                                  'Left Subarticular Stenosis',
                                                  'Right Subarticular Stenosis'],
                                      series_description='Sagittal T2/STIR',
                                      mode='train', 
                                      transform=aug.get_transform(cfg=CFG, train=True))
    
    # print(ds.label_columns)
    # print(ds.available_diagnosis)
    x, t = ds[0]
    assert x.shape == torch.Size([5, CH, 2*SS, 2*SS])
    assert t['labels'].shape == torch.Size([25])
    assert t['study_id'] == torch.tensor([4003253])


def test_dataset_sagittalt1():
    CH = 7
    SS = 64
    ds = level_cubes.LevelCubeDataset(study_ids=[4003253,    4646740,    7143189],
                                      channels=CH,
                                      slice_size=SS,
                                      conditions=['Spinal Canal Stenosis',     
                                                  'Left Neural Foraminal Narrowing', 
                                                  'Right Neural Foraminal Narrowing',
                                                  'Left Subarticular Stenosis',
                                                  'Right Subarticular Stenosis'],
                                      series_description='Sagittal T1',
                                      mode='train', 
                                      transform=aug.get_transform(cfg=CFG, train=True))
    
    x, t = ds[1]
    assert x.shape == torch.Size([5, CH, 2*SS, 2*SS])
    assert t['labels'].shape == torch.Size([25])
    assert t['study_id'] == torch.tensor([4646740])


def test_dataset_axialt2():
    CH = 7
    SS = 64
    ds = level_cubes.LevelCubeDataset(study_ids=[4003253,    4646740,    7143189],
                                      channels=CH,
                                      slice_size=SS,
                                      conditions=['Spinal Canal Stenosis',     
                                                  'Left Neural Foraminal Narrowing', 
                                                  'Right Neural Foraminal Narrowing',
                                                  'Left Subarticular Stenosis',
                                                  'Right Subarticular Stenosis'],
                                      series_description='Axial T2',
                                      mode='train', 
                                      transform=aug.get_transform(cfg=CFG, train=True))
    
    x, t = ds[2]
    assert x.shape == torch.Size([5, CH, 2*SS, 2*SS])
    assert t['labels'].shape == torch.Size([25])
    assert t['study_id'] == torch.tensor([7143189])


def test_dataset_all():
    CH = 7
    SS = 64
    ds = level_cubes.AllLevelCubeDataset(study_ids=[4003253,    4646740,    7143189],
                                      channels=CH,
                                      slice_size=SS,
                                      conditions=['Spinal Canal Stenosis',     
                                                  'Left Neural Foraminal Narrowing', 
                                                  'Right Neural Foraminal Narrowing',
                                                  'Left Subarticular Stenosis',
                                                  'Right Subarticular Stenosis'],
                                      mode='train', 
                                      transform=aug.get_transform(cfg=CFG, train=True))
    
    (x1, x2, x3), t = ds[0]
    assert x1.shape == torch.Size([5, CH, 2*SS, 2*SS])
    assert x2.shape == torch.Size([5, CH, 2*SS, 2*SS])
    assert x3.shape == torch.Size([5, CH, 2*SS, 2*SS])
    assert t['labels'].shape == torch.Size([25])
    assert t['study_id'] == torch.tensor([4003253])


def test_dataloader():
    CH = 7
    SS = 64
    ds = level_cubes.AllLevelCubeDataset(study_ids=[4003253,    4646740,    7143189],
                                      channels=CH,
                                      slice_size=SS,
                                      conditions=['Spinal Canal Stenosis',     
                                                  'Left Neural Foraminal Narrowing', 
                                                  'Right Neural Foraminal Narrowing',
                                                  'Left Subarticular Stenosis',
                                                  'Right Subarticular Stenosis'],
                                      mode='train', 
                                      transform=None)
    
    dl = DataLoader(ds, batch_size=3, shuffle=False, pin_memory=True, drop_last=True, num_workers=1)
    
    (x1, x2, x3), t = next(iter(dl))

    assert x1.shape == torch.Size([3, 5, CH, 2*SS, 2*SS])
    assert x2.shape == torch.Size([3, 5, CH, 2*SS, 2*SS])
    assert x3.shape == torch.Size([3, 5, CH, 2*SS, 2*SS])
    assert t['labels'].shape == torch.Size([3, 25])
    assert torch.all(t['study_id'] == torch.tensor([[4003253], [4646740], [7143189]]))