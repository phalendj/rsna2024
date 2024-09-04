import torch
import pytest

from torch.utils.data import DataLoader

import datasets
import datasets.full_level as full_level
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


def test_dataset():
    rsnautils.set_clean(False)
    CH_A = 7
    SS_A = 64
    CH_S = 35
    SS_S = 80
    ds = full_level.FullLevelDataset(study_ids=[4003253,    4646740,    7143189],
                                      channels_sag=CH_S,
                                      patch_size_sag=SS_S,
                                      d_sag=50,
                                      d_slice_sag=3,
                                      channels_ax=CH_A,
                                      patch_size_ax=SS_A,
                                      d_ax=80,
                                      d_slice_ax=3,
                                      conditions=datasets.CONDITIONS,
                                      aug_size=0.0,
                                      generated_coordinate_file='/data/phalendj/kaggle/rsna2024/train_label_coordinates.csv',
                                      mode='train', 
                                      transform=aug.get_transform(cfg=CFG, train=True))
    
    data, targets = ds[0]
    # print('Data')
    # for k, v in data.items():
    #     print(k, v.shape)

    

    assert data['series_ids'].shape == torch.Size([5, 3])

    assert data['Sagittal T2/STIR Patch'].shape == torch.Size([5, CH_S, SS_S, SS_S])
    assert data['Sagittal T2/STIR Instance Numbers'].shape == torch.Size([5, CH_S])
    assert data['Sagittal T2/STIR Offsets'].shape == torch.Size([5, 2])

    assert data['Sagittal T1 Patch'].shape == torch.Size([5, CH_S, SS_S, SS_S])
    assert data['Sagittal T1 Instance Numbers'].shape == torch.Size([5, CH_S])
    assert data['Sagittal T1 Offsets'].shape == torch.Size([5, 2])

    assert data['Axial T2 Patch'].shape == torch.Size([5, CH_A, SS_A, SS_A])
    assert data['Axial T2 Instance Numbers'].shape == torch.Size([5, CH_A])
    assert data['Axial T2 Offsets'].shape == torch.Size([5, 2])

    # print('targets')
    # for k, v in targets.items():
    #     print(k, v.shape)
    assert targets['labels'].shape == torch.Size([25])
    assert targets['study_id'] == torch.tensor([4003253])
    assert targets['Spinal Canal Stenosis Centers'].shape == torch.Size([5, 2])
    assert targets['Left Neural Foraminal Narrowing Centers'].shape == torch.Size([5, 2])
    assert targets['Right Neural Foraminal Narrowing Centers'].shape == torch.Size([5, 2])
    assert targets['Left Subarticular Stenosis Centers'].shape == torch.Size([5, 2])
    assert targets['Left Subarticular Stenosis Centers'].shape == torch.Size([5, 2])

    assert targets['Spinal Canal Stenosis Slice Classification'].shape == torch.Size([5, CH_S])
    assert targets['Left Neural Foraminal Narrowing Slice Classification'].shape == torch.Size([5, CH_S])
    assert targets['Right Neural Foraminal Narrowing Slice Classification'].shape == torch.Size([5, CH_S])
    assert targets['Left Subarticular Stenosis Slice Classification'].shape == torch.Size([5, CH_A])
    assert targets['Left Subarticular Stenosis Slice Classification'].shape == torch.Size([5, CH_A])
    


def test_dataloader():
    rsnautils.set_clean(False)
    CH_A = 7
    SS_A = 64
    CH_S = 15
    SS_S = 80
    ds = full_level.FullLevelDataset(study_ids=[4003253,    4646740,    7143189],
                                      channels_sag=CH_S,
                                      patch_size_sag=SS_S,
                                      d_sag=50,
                                      d_slice_sag=3,
                                      channels_ax=CH_A,
                                      patch_size_ax=SS_A,
                                      d_ax=80,
                                      d_slice_ax=3,
                                      conditions=datasets.CONDITIONS,
                                      aug_size=0.0,
                                      generated_coordinate_file='/data/phalendj/kaggle/rsna2024/train_label_coordinates.csv',
                                      mode='train', 
                                      transform=aug.get_transform(cfg=CFG, train=True))
    
    dl = DataLoader(ds, batch_size=3, shuffle=False, pin_memory=True, drop_last=True, num_workers=1)
    data, targets = next(iter(dl))
    # print('Data')
    # for k, v in data.items():
    #     print(k, v.shape)

    assert data['series_ids'].shape == torch.Size([3, 5, 3])

    assert data['Sagittal T2/STIR Patch'].shape == torch.Size([3, 5, CH_S, SS_S, SS_S])
    assert data['Sagittal T2/STIR Instance Numbers'].shape == torch.Size([3, 5, CH_S])
    assert data['Sagittal T2/STIR Offsets'].shape == torch.Size([3, 5, 2])

    assert data['Sagittal T1 Patch'].shape == torch.Size([3, 5, CH_S, SS_S, SS_S])
    assert data['Sagittal T1 Instance Numbers'].shape == torch.Size([3, 5, CH_S])
    assert data['Sagittal T1 Offsets'].shape == torch.Size([3, 5, 2])

    assert data['Axial T2 Patch'].shape == torch.Size([3, 5, CH_A, SS_A, SS_A])
    assert data['Axial T2 Instance Numbers'].shape == torch.Size([3, 5, CH_A])
    assert data['Axial T2 Offsets'].shape == torch.Size([3, 5, 2])

    # print('targets')
    # for k, v in targets.items():
    #     print(k, v.shape)
    assert targets['labels'].shape == torch.Size([3, 25])
    assert torch.all(targets['study_id'] == torch.tensor([[4003253], [4646740], [7143189]]))
    assert targets['Spinal Canal Stenosis Centers'].shape == torch.Size([3, 5, 2])
    assert targets['Left Neural Foraminal Narrowing Centers'].shape == torch.Size([3, 5, 2])
    assert targets['Right Neural Foraminal Narrowing Centers'].shape == torch.Size([3, 5, 2])
    assert targets['Left Subarticular Stenosis Centers'].shape == torch.Size([3, 5, 2])
    assert targets['Left Subarticular Stenosis Centers'].shape == torch.Size([3, 5, 2])

    assert targets['Spinal Canal Stenosis Slice Classification'].shape == torch.Size([3, 5, CH_S])
    assert targets['Left Neural Foraminal Narrowing Slice Classification'].shape == torch.Size([3, 5, CH_S])
    assert targets['Right Neural Foraminal Narrowing Slice Classification'].shape == torch.Size([3, 5, CH_S])
    assert targets['Left Subarticular Stenosis Slice Classification'].shape == torch.Size([3, 5, CH_A])
    assert targets['Left Subarticular Stenosis Slice Classification'].shape == torch.Size([3, 5, CH_A])

    
    