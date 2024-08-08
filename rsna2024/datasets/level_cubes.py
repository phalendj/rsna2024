import logging
import numpy as np
import random
import math

import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision

try:
    from utils import relative_directory
    from datasets import load_train_files, load_test_files, LEVELS, CONDITIONS, create_column
    from datasets.dicom_load import Study
except ImportError:
    from ..utils import relative_directory
    from ..datasets import load_train_files, load_test_files, LEVELS, CONDITIONS, create_column
    from .dicom_load import Study

logger = logging.getLogger(__name__)



class LevelCubeDataset(Dataset):
    def __init__(self, study_ids, channels: int, slice_size: int, series_description, conditions, mode='train', transform=None, load_studies=None):
        self.study_ids = list(study_ids)
        self.slice_size = int(slice_size)
        self.channels = channels
        logger.info(f'Output will have size {2*self.slice_size} and {channels} channels')
        self.mode = mode
        self.transform = transform
        if self.mode == 'train' or self.mode == 'valid':
            self.labels_df, self.coordinate_df, self.series_description_df = load_train_files(relative_directory=relative_directory)
        else:
            self.labels_df = None
            self.coordinate_df = None
            self.series_description_df = load_test_files(relative_directory=relative_directory)

        if load_studies is None:
            logger.info(f'Loading {len(study_ids)} Studies')
            self.studies = [Study(study_id=study_id, labels_df=self.labels_df, series_description_df=self.series_description_df, coordinate_df=self.coordinate_df) for study_id in study_ids]
            logger.info(f'Done')
        else:
            logger.info(f'Referencing pre-loaded studies')
            self.studies = load_studies
            assert len(self.study_ids) == len(self.studies)


        self.label_columns = sum([[create_column(condition, level=level) for level in LEVELS] for condition in CONDITIONS if condition in conditions], [])
        self.series_description = series_description

        if self.mode == 'train' or self.mode == 'valid':
            series2cond = {'Sagittal T2/STIR': 'spinal',  'Sagittal T1': 'foraminal', 'Axial T2': 'subarticular'}
            self.available_diagnosis = [c for c in self.label_columns if series2cond[self.series_description] in c]
            

    @property
    def labels(self):
        return self.label_columns

    def __getitem__(self, idx):
        """
        Will output a tensor of size LEVELS, CHANNELS, 2*SLICE_SIZE, SLICE_SIZE
        """
        final_size = 5, 2*self.slice_size, 2*self.slice_size, int(self.channels)
        x = np.zeros(final_size, dtype=np.uint8)
        study = self.studies[idx]
        if self.mode == 'train' or self.mode == 'valid':
            label = np.int64([study.labels[col] for col in self.label_columns])
        else:
            label = np.int64([-100 for col in self.label_columns])

        available = [s[2] for s in study.series if s[1] == self.series_description]
        if len(available) > 0:
            if self.mode == 'train':
                series = np.random.choice(available)
            else:
                series = available[0]
            
            data = series.data
            data = data.transpose(1, 2, 0)
            H, W, D = data.shape

            for k in range(5):
                xr = []
                yr = []
                zr = []
                k2 = k
                while k2 < len(self.available_diagnosis):

                    nm = self.available_diagnosis[k2]
                    if nm in series.diagnosis_coordinates:
                        y0, x0, z0 = series.diagnosis_coordinates[nm]
                        xr.append(x0)
                        yr.append(y0)
                        zr.append(z0)
                    k2 += 5

                if len(xr) > 0:
                    x0 = np.mean(xr)
                    y0 = np.mean(yr)
                    z0 = np.mean(zr)
                    x0 = min(max(int(x0) - self.slice_size, 0), H - 2*self.slice_size)
                    y0 = min(max(int(y0) - self.slice_size, 0), W - 2*self.slice_size)
                    x1 = min(H, x0 + 2*self.slice_size)
                    y1 = min(W, y0 + 2*self.slice_size)

                    i0 = max(int(z0 - self.channels/ 2), 0)
                    i1 = min(i0 + self.channels, D)
                    # Axial images can have smaller pixel sizes, and when doing levels we really want the entire image
                    if self.series_description == 'Axial T2' and (2*self.slice_size > H or 2*self.slice_size > W):
                        data2 = data[..., i0:i1].copy()
                        if H > W:
                            diff = H-W
                            if self.mode == 'train':
                                offset = np.random.randint(diff)
                            else:
                                offset = int(diff//2)
                            data2 = data2[offset:offset+W]
                            H = W
                        elif W > H:
                            diff = W-H
                            if self.mode == 'train':
                                offset = np.random.randint(diff)
                            else:
                                offset = int(diff//2)

                            data2 = data2[:, offset:offset+H]
                            W = H
                        data2 = cv2.resize(data2, (2*self.slice_size, 2*self.slice_size), interpolation=cv2.INTER_LANCZOS4)
                        x[k, ..., :(i1-i0)] = data2
                        
                    else:
                        x[k, :(x1-x0), :(y1-y0), :(i1-i0)] = data[x0:x1, y0:y1, i0:i1]
                    
            if self.transform is not None:
                # Need to reshape it around
                x = x.transpose(1, 2, 3, 0).reshape(2*self.slice_size, 2*self.slice_size, -1)
                x = self.transform(image=x)['image'].reshape(2*self.slice_size, 2*self.slice_size, self.channels, -1).transpose(3, 0, 1, 2)
        
        x = x.transpose(0, 3, 1, 2)

        target = {}
        target['labels'] = torch.tensor(label)
        target['study_id'] = torch.tensor([study.study_id])

        return torch.tensor(x, dtype=torch.float) / 255.0, target

    def __len__(self):
        return len(self.study_ids)
    

class AllLevelCubeDataset(Dataset):
    def __init__(self, study_ids, channels: int, slice_size: int, conditions, mode='train', transform=None):
        self.sagittal_t2 = LevelCubeDataset(study_ids=study_ids, channels=channels, slice_size=slice_size, conditions=conditions, mode=mode, transform=transform, series_description='Sagittal T2/STIR')
        self.sagittal_t1 = LevelCubeDataset(study_ids=study_ids, channels=channels, slice_size=slice_size, conditions=conditions, mode=mode, transform=transform, series_description='Sagittal T1', load_studies=self.sagittal_t2.studies)
        self.axial_t2 = LevelCubeDataset(study_ids=study_ids, channels=channels, slice_size=slice_size, conditions=conditions, mode=mode, transform=transform, series_description='Axial T2', load_studies=self.sagittal_t2.studies)

    def __getitem__(self, idx):
        x_st2, target = self.sagittal_t2[idx]
        x_st1, target = self.sagittal_t1[idx]
        x_ax2, target = self.axial_t2[idx]

        return (x_st2, x_st1, x_ax2), target

    @property
    def labels(self):
        return self.sagittal_t2.labels

    def __len__(self):
        return len(self.sagittal_t2)
    


class LevelCubeLeftRightDataset(Dataset):
    def __init__(self, study_ids, channels: int, slice_size: int, series_description, conditions, mode='train', transform=None, load_studies=None):
        self.study_ids = list(study_ids)
        self.slice_size = int(slice_size)
        self.channels = channels
        logger.info(f'Output will have size {2*self.slice_size} and {channels} channels')
        assert len(conditions) == 2, "Conditions must be same style"
        self.mode = mode
        self.transform = transform
        if self.mode == 'train' or self.mode == 'valid':
            self.labels_df, self.coordinate_df, self.series_description_df = load_train_files(relative_directory=relative_directory)
        else:
            self.labels_df = None
            self.coordinate_df = None
            self.series_description_df = load_test_files(relative_directory=relative_directory)

        if load_studies is None:
            logger.info(f'Loading {len(study_ids)} Studies')
            self.studies = [Study(study_id=study_id, labels_df=self.labels_df, series_description_df=self.series_description_df, coordinate_df=self.coordinate_df) for study_id in study_ids]
            logger.info(f'Done')
        else:
            logger.info(f'Referencing pre-loaded studies')
            self.studies = load_studies
            assert len(self.study_ids) == len(self.studies)


        self.label_columns = sum([[create_column(condition, level=level) for level in LEVELS] for condition in CONDITIONS if condition in conditions], [])
        assert len(self.label_columns) == 10
        # assert np.all(['foraminal' in c for c in self.label_columns])
        self.series_description = series_description

        if self.mode == 'train' or self.mode == 'valid':
            series2cond = {'Sagittal T2/STIR': 'spinal',  'Sagittal T1': 'foraminal', 'Axial T2': 'subarticular'}
            self.available_diagnosis = [c for c in self.label_columns if series2cond[self.series_description] in c]
            
    @property
    def labels(self):
        return self.label_columns

    def __getitem__(self, idx):
        """
        Will output a tensor of size LEVELS, SIDE, CHANNELS, 2*SLICE_SIZE, 2*SLICE_SIZE
        """
        final_size = 5, 2, 2*self.slice_size, 2*self.slice_size, int(self.channels)
        x = np.zeros(final_size, dtype=np.uint8)
        study = self.studies[idx]
        if self.mode == 'train' or self.mode == 'valid':
            label = np.int64([study.labels[col] for col in self.label_columns])
        else:
            label = np.int64([-100 for col in self.label_columns])

        available = [s[2] for s in study.series if s[1] == self.series_description]
        if len(available) > 0:
            if self.mode == 'train':
                series = np.random.choice(available)
            else:
                series = available[0]
            
            data = series.data
            data = data.transpose(1, 2, 0)
            H, W, D = data.shape

            for k in range(5):
                for m, side in enumerate(['left', 'right']):
                    xr = []
                    yr = []
                    zr = []
                    k2 = k
                    while k2 < len(self.available_diagnosis):

                        nm = self.available_diagnosis[k2]

                        if nm in series.diagnosis_coordinates and side in nm:
                            y0, x0, z0 = series.diagnosis_coordinates[nm]
                            xr.append(x0)
                            yr.append(y0)
                            zr.append(z0)
                        k2 += 5

                    if len(xr) > 0:
                        x0 = np.mean(xr)
                        y0 = np.mean(yr)
                        z0 = np.mean(zr)
                        x0 = min(max(int(x0) - self.slice_size, 0), H - 2*self.slice_size)
                        y0 = min(max(int(y0) - self.slice_size, 0), W - 2*self.slice_size)
                        x1 = min(H, x0 + 2*self.slice_size)
                        y1 = min(W, y0 + 2*self.slice_size)

                        i0 = max(int(z0 - self.channels/ 2), 0)
                        i1 = min(i0 + self.channels, D)
                        # Axial images can have smaller pixel sizes, and when doing levels we really want the entire image
                        if self.series_description == 'Axial T2' and (2*self.slice_size > H or 2*self.slice_size > W):
                            data2 = data[..., i0:i1].copy()
                            if H > W:
                                diff = H-W
                                if self.mode == 'train':
                                    offset = np.random.randint(diff)
                                else:
                                    offset = int(diff//2)
                                data2 = data2[offset:offset+W]
                                H = W
                            elif W > H:
                                diff = W-H
                                if self.mode == 'train':
                                    offset = np.random.randint(diff)
                                else:
                                    offset = int(diff//2)

                                data2 = data2[:, offset:offset+H]
                                W = H
                            data2 = cv2.resize(data2, (2*self.slice_size, 2*self.slice_size), interpolation=cv2.INTER_LANCZOS4)
                            x[k, ..., :(i1-i0)] = data2
                            
                        else:
                            x[k, m, :(x1-x0), :(y1-y0), :(i1-i0)] = data[x0:x1, y0:y1, i0:i1]
                    
            if self.transform is not None:
                # Need to reshape it around
                x = x.transpose(2, 3, 4, 0, 1).reshape(2*self.slice_size, 2*self.slice_size, -1)
                x = self.transform(image=x)['image'].reshape(2*self.slice_size, 2*self.slice_size, self.channels, 5, 2).transpose(3, 4, 0, 1, 2)
        
        x = x.transpose(0, 1, 4, 2, 3)

        target = {}
        target['labels'] = torch.tensor(label)
        target['study_id'] = torch.tensor([study.study_id])

        return torch.tensor(x, dtype=torch.float) / 255.0, target

    def __len__(self):
        return len(self.study_ids)


class LevelCubeCropZoomDataset(Dataset):
    def __init__(self, study_ids, channels: int, image_size: tuple[int, int], slice_size: int, series_description, conditions, mode='train', transform=None, load_studies=None):
        self.study_ids = list(study_ids)
        self.slice_size = int(slice_size)
        self.image_size = int(image_size[0]), int(image_size[1])
        self.channels = channels
        logger.info(f'Output will have size {2*self.slice_size} resized to {image_size} and {channels} channels')
        self.mode = mode
        self.transform = transform
        if self.mode == 'train' or self.mode == 'valid':
            self.labels_df, self.coordinate_df, self.series_description_df = load_train_files(relative_directory=relative_directory)
        else:
            self.labels_df = None
            self.coordinate_df = None
            self.series_description_df = load_test_files(relative_directory=relative_directory)

        if load_studies is None:
            logger.info(f'Loading {len(study_ids)} Studies')
            self.studies = [Study(study_id=study_id, labels_df=self.labels_df, series_description_df=self.series_description_df, coordinate_df=self.coordinate_df) for study_id in study_ids]
            logger.info(f'Done')
        else:
            logger.info(f'Referencing pre-loaded studies')
            self.studies = load_studies
            assert len(self.study_ids) == len(self.studies)


        self.label_columns = sum([[create_column(condition, level=level) for level in LEVELS] for condition in CONDITIONS if condition in conditions], [])
        self.series_description = series_description

        if self.mode == 'train' or self.mode == 'valid':
            series2cond = {'Sagittal T2/STIR': 'spinal',  'Sagittal T1': 'foraminal', 'Axial T2': 'subarticular'}
            self.available_diagnosis = [c for c in self.label_columns if series2cond[self.series_description] in c]
            

    @property
    def labels(self):
        return self.label_columns

    def __getitem__(self, idx):
        """
        Will output a tensor of size LEVELS, CHANNELS, 2*SLICE_SIZE, SLICE_SIZE
        """
        final_size = 5, self.image_size[0], self.image_size[1], int(self.channels)
        x = np.zeros(final_size, dtype=np.uint8)
        study = self.studies[idx]
        if self.mode == 'train' or self.mode == 'valid':
            label = np.int64([study.labels[col] for col in self.label_columns])
        else:
            label = np.int64([-100 for col in self.label_columns])

        available = [s[2] for s in study.series if s[1] == self.series_description]
        if len(available) > 0:
            if self.mode == 'train':
                series = np.random.choice(available)
            else:
                series = available[0]
            
            data = series.data
            data = data.transpose(1, 2, 0)
            H, W, D = data.shape

            for k in range(5):
                xr = []
                yr = []
                zr = []
                k2 = k
                while k2 < len(self.available_diagnosis):

                    nm = self.available_diagnosis[k2]
                    if nm in series.diagnosis_coordinates:
                        y0, x0, z0 = series.diagnosis_coordinates[nm]
                        xr.append(x0)
                        yr.append(y0)
                        zr.append(z0)
                    k2 += 5

                if len(xr) > 0:
                    x0 = np.mean(xr)
                    y0 = np.mean(yr)
                    z0 = np.mean(zr)
                    x0 = min(max(int(x0) - self.slice_size, 0), H - 2*self.slice_size)
                    y0 = min(max(int(y0) - self.slice_size, 0), W - 2*self.slice_size)
                    x1 = min(H, x0 + 2*self.slice_size)
                    y1 = min(W, y0 + 2*self.slice_size)

                    i0 = max(int(z0 - self.channels/ 2), 0)
                    i1 = min(i0 + self.channels, D)
                    # Axial images can have smaller pixel sizes, and when doing levels we really want the entire image
                    if self.series_description == 'Axial T2' and (2*self.slice_size > H or 2*self.slice_size > W):
                        data2 = data[..., i0:i1].copy()
                        if H > W:
                            diff = H-W
                            if self.mode == 'train':
                                offset = np.random.randint(diff)
                            else:
                                offset = int(diff//2)
                            data2 = data2[offset:offset+W]
                            H = W
                        elif W > H:
                            diff = W-H
                            if self.mode == 'train':
                                offset = np.random.randint(diff)
                            else:
                                offset = int(diff//2)

                            data2 = data2[:, offset:offset+H]
                            W = H
                        data2 = cv2.resize(data2, self.image_size, interpolation=cv2.INTER_LANCZOS4)
                        x[k, ..., :(i1-i0)] = data2
                        
                    else:
                        x[k, :, :, :(i1-i0)] = cv2.resize(data[x0:x1, y0:y1, i0:i1], self.image_size, interpolation=cv2.INTER_LANCZOS4)
                    
            if self.transform is not None:
                # Need to reshape it around
                x = x.transpose(1, 2, 3, 0).reshape(*self.image_size, -1)
                x = self.transform(image=x)['image'].reshape(*self.image_size, self.channels, -1).transpose(3, 0, 1, 2)
        
        x = x.transpose(0, 3, 1, 2)

        target = {}
        target['labels'] = torch.tensor(label)
        target['study_id'] = torch.tensor([study.study_id])

        return torch.tensor(x, dtype=torch.float) / 255.0, target

    def __len__(self):
        return len(self.study_ids)
