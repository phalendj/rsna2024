import torch
import pytest

from torch.utils.data import DataLoader

import datasets.level_cubes as level_cubes
import datasets.augmentations as aug
import utils as rsnautils


class CROP:
    use = False
    size = 512

class CFG:
    hflip = False
    contrast = True
    blur = True
    blur_limit = 9
    noise = True
    downscale = True
    distort = True
    rotate = True
    normalize=True
    prob = 0.75
    sharpen = False
    channel_shuffle = False
    crop = CROP


def test_dataset_sagittalt2():
    rsnautils.set_clean(False)
    CH = 7
    SS = 64
    ds = level_cubes.LevelCubeDataset(study_ids=[4003253,    4646740,    7143189],
                                      channels=CH,
                                      patch_size=SS,
                                      condition='Spinal Canal Stenosis',
                                      series_description='Sagittal T2/STIR',
                                      generated_coordinate_file='/data/phalendj/kaggle/rsna2024/train_label_coordinates.csv',
                                      mode='train', 
                                      transform=aug.get_transform(cfg=CFG, train=True))
    
    x, t = ds[0]
    assert x.shape == torch.Size([5, CH, SS, SS])
    assert t['labels'].shape == torch.Size([5])
    assert t['centers'].shape == torch.Size([5, 2])
    assert t['patch_offsets'].shape == torch.Size([5, 2])
    assert t['study_id'] == torch.tensor([4003253])
    assert t['slice_classification'].sum().item() == 5


def test_dataset_sagittalt1():
    rsnautils.set_clean(False)
    CH = 7
    SS = 64
    ds = level_cubes.LevelCubeDataset(study_ids=[4003253,    4646740,    7143189],
                                      channels=CH,
                                      patch_size=SS,
                                      condition='Left Neural Foraminal Narrowing',
                                      series_description='Sagittal T1',
                                      generated_coordinate_file='/data/phalendj/kaggle/rsna2024/train_label_coordinates.csv',
                                      mode='train', 
                                      transform=aug.get_transform(cfg=CFG, train=True))
    
    x, t = ds[1]
    assert x.shape == torch.Size([5, CH, SS, SS])
    assert t['labels'].shape == torch.Size([5])
    assert t['centers'].shape == torch.Size([5, 2])
    assert t['patch_offsets'].shape == torch.Size([5, 2])
    assert t['study_id'] == torch.tensor([4646740])
    assert t['slice_classification'].sum().item() == 5


def test_dataset_axialt2():
    rsnautils.set_clean(False)
    CH = 7
    SS = 64
    ds = level_cubes.LevelCubeDataset(study_ids=[4003253,    4646740,    7143189],
                                      channels=CH,
                                      patch_size=SS,
                                      condition='Left Subarticular Stenosis',
                                      series_description='Axial T2',
                                      generated_coordinate_file='/data/phalendj/kaggle/rsna2024/train_label_coordinates.csv',
                                      mode='train', 
                                      transform=aug.get_transform(cfg=CFG, train=True))
    
    x, t = ds[2]
    assert x.shape == torch.Size([5, CH, SS, SS])
    assert t['labels'].shape == torch.Size([5])
    assert t['centers'].shape == torch.Size([5, 2])
    assert t['patch_offsets'].shape == torch.Size([5, 2])
    assert t['study_id'] == torch.tensor([7143189])
    assert t['slice_classification'].sum().item() == 5


def test_all_dataset_axialt2():
    rsnautils.set_clean(False)
    CH1 = 7
    SS1 = 64
    CH2 = 5
    SS2 = 48
    CH3 = 3
    SS3 = 72
    ds = level_cubes.AllLevelCubeDataset(study_ids=[4003253,    4646740,    7143189],
                                      sagittal_t2_channels=CH1,
                                      sagittal_t1_channels=CH2,
                                      axial_t2_channels=CH3,
                                      sagittal_t2_patch_size=SS1,
                                      sagittal_t1_patch_size=SS2,
                                      axial_t2_patch_size=SS3,
                                      condition='Left Subarticular Stenosis',
                                      series_description='Axial T2',
                                      generated_coordinate_file='/data/phalendj/kaggle/rsna2024/train_label_coordinates.csv',
                                      mode='train', 
                                      transform=aug.get_transform(cfg=CFG, train=True))
    
    (x1, x2, x3), t = ds[2]
    assert x1.shape == torch.Size([5, CH1, SS1, SS1])
    assert x2.shape == torch.Size([5, CH2, SS2, SS2])
    assert x3.shape == torch.Size([5, CH3, SS3, SS3])
    assert t['labels'].shape == torch.Size([5])
    assert t['study_id'] == torch.tensor([7143189])
    

# def test_dataset_all():
#     CH = 7
#     SS = 64
#     ds = level_cubes.AllLevelCubeDataset(study_ids=[4003253,    4646740,    7143189],
#                                       channels=CH,
#                                       slice_size=SS,
#                                       conditions=['Spinal Canal Stenosis',     
#                                                   'Left Neural Foraminal Narrowing', 
#                                                   'Right Neural Foraminal Narrowing',
#                                                   'Left Subarticular Stenosis',
#                                                   'Right Subarticular Stenosis'],
#                                       mode='train', 
#                                       transform=aug.get_transform(cfg=CFG, train=True))
    
#     (x1, x2, x3), t = ds[0]
#     assert x1.shape == torch.Size([5, CH, 2*SS, 2*SS])
#     assert x2.shape == torch.Size([5, CH, 2*SS, 2*SS])
#     assert x3.shape == torch.Size([5, CH, 2*SS, 2*SS])
#     assert t['labels'].shape == torch.Size([25])
#     assert t['study_id'] == torch.tensor([4003253])


# def test_dataloader():
#     CH = 7
#     SS = 64
#     ds = level_cubes.AllLevelCubeDataset(study_ids=[4003253,    4646740,    7143189],
#                                       channels=CH,
#                                       slice_size=SS,
#                                       conditions=['Spinal Canal Stenosis',     
#                                                   'Left Neural Foraminal Narrowing', 
#                                                   'Right Neural Foraminal Narrowing',
#                                                   'Left Subarticular Stenosis',
#                                                   'Right Subarticular Stenosis'],
#                                       mode='train', 
#                                       transform=None)
    
#     dl = DataLoader(ds, batch_size=3, shuffle=False, pin_memory=True, drop_last=True, num_workers=1)
    
#     (x1, x2, x3), t = next(iter(dl))

#     assert x1.shape == torch.Size([3, 5, CH, 2*SS, 2*SS])
#     assert x2.shape == torch.Size([3, 5, CH, 2*SS, 2*SS])
#     assert x3.shape == torch.Size([3, 5, CH, 2*SS, 2*SS])
#     assert t['labels'].shape == torch.Size([3, 25])
#     assert torch.all(t['study_id'] == torch.tensor([[4003253], [4646740], [7143189]]))