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


def rotate_image(image, angle, interpolation=cv2.INTER_LANCZOS4):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=interpolation)
    return result


def augment_image_and_centers(image,centers,alpha):
    '''
    # Randomly flip the image horizontally.
    if random.random() > .5:
      if random.random() > 1 - alpha:
        image = image.flip(-1)
        centers[:,0] = PATCH_SIZE - centers[:,0]
    # Randomly flip the image vertically.
    if random.random() > 0.5:
      if random.random() > 1 - alpha:
        image = image.flip(-2)
        centers[:,1] = PATCH_SIZE - centers[:,1]
  
    if random.random() > 1 - alpha:
      if random.random() > .5:
    #   Randomly flip the image
    #   Wich axis?
        axis = np.random.randint(2)
        image = image.flip(axis+1)
        centers[:,-1-axis] = PATCH_SIZE - centers[:,-1-axis]
    '''
#   Randomly rotate the image.
    shp = image.shape
    if len(shp) == 3:
        H, W, D = shp
    else:
        H, W = shp
        D = 1
    assert H == W
    PATCH_SIZE = H

    angle = random.uniform(-180, 180)*alpha
    image = rotate_image(image, angle, interpolation=cv2.INTER_LANCZOS4 if D <= 3 else cv2.INTER_LINEAR)
#   https://discuss.pytorch.org/t/rotation-matrix/128260
    angle = torch.tensor(-angle*math.pi/180)
    s = torch.sin(angle)
    c = torch.cos(angle)
    rot = torch.stack([
        torch.stack([c, s]),
        torch.stack([-s, c])
      ])
    centers = ((centers.cpu() - PATCH_SIZE//2) @ rot) + PATCH_SIZE//2

    return image, centers



class SegmentationSingleImageDataset(Dataset):
    def __init__(self, study_ids, image_size, series_description, conditions, mode='train', aug_size=0.0):
        self.study_ids = list(study_ids)
        self.image_size = int(image_size[0]), int(image_size[1])
        logger.info(f'Output will have size {self.image_size}')
        self.mode = mode
        self.aug_size = aug_size
        if self.mode == 'train' or self.mode == 'valid':
            self.labels_df, self.coordinate_df, self.series_description_df = load_train_files(relative_directory=relative_directory)
        else:
            self.labels_df = None
            self.coordinate_df = None
            self.series_description_df = load_test_files(relative_directory=relative_directory)

        logger.info(f'Loading {len(study_ids)} Studies')
        self.studies = [Study(study_id=study_id, labels_df=self.labels_df, series_description_df=self.series_description_df, coordinate_df=self.coordinate_df) for study_id in study_ids]
        logger.info(f'Done')

        self.label_columns = sum([[create_column(condition, level=level) for level in LEVELS] for condition in CONDITIONS if condition in conditions], [])
        self.series_description = series_description

        if self.mode == 'train' or self.mode == 'valid':
            series2cond = {'Sagittal T2/STIR': 'spinal',  'Sagittal T1': 'foraminal', 'Axial T2': 'subarticular'}
            self.available_diagnosis = [c for c in self.label_columns if series2cond[self.series_description] in c]
            

    @property
    def labels(self):
        return self.label_columns

    def __getitem__(self, idx):
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

            centers = np.array([list(series.diagnosis_coordinates[k]) if k in series.diagnosis_coordinates else [-1.e4, -1.e4, -1] for k in self.available_diagnosis])
            slice = int(np.random.choice([c for c in centers[:, 2] if c >=0]))
            centers = torch.as_tensor(centers[:, :2]).float()

            data = series.data[slice]

            H, W = data.shape
            if H > W:
                diff = H-W
                if self.mode == 'train':
                    offset = np.random.randint(diff)
                else:
                    offset = int(diff//2)
                data = data[offset:offset+W]
                centers[:,1] -= offset
                H = W
            elif W > H:
                diff = W-H
                if self.mode == 'train':
                    offset = np.random.randint(diff)
                else:
                    offset = int(diff//2)

                data = data[:, offset:offset+H]
                centers[:,0] -= offset
                W = H

            # logger.info(f'Data Shape : {data.shape}')
            data = cv2.resize(data, self.image_size, interpolation=cv2.INTER_LANCZOS4)

            centers[:,0] = centers[:,0]*self.image_size[0]/W
            centers[:,1] = centers[:,1]*self.image_size[1]/H

            if self.mode == 'train':
                data, centers = augment_image_and_centers(image=data, centers=centers, alpha=self.aug_size)
        else:
            data = np.zeros(self.image_size, dtype=float)
            centers = torch.as_tensor([[-1.e4, -1.e4] for k in self.available_diagnosis]).float()

        target = {}
        target['labels'] = torch.tensor(label)
        target['centers'] = centers
        target['study_id'] = torch.tensor([study.study_id])

        return torch.tensor(data, dtype=torch.float).unsqueeze(0) / 255.0, target

    def __len__(self):
        return len(self.study_ids)
    


class SegmentationCenterDataset(Dataset):
    def __init__(self, study_ids, image_size, channels: int, series_description, conditions, mode='train', aug_size=0.0, transform=None):
        self.study_ids = list(study_ids)
        self.image_size = int(image_size[0]), int(image_size[1])
        self.channels = channels
        logger.info(f'Output will have size {self.image_size} and {channels} channels')
        self.mode = mode
        self.aug_size = aug_size
        self.transform = transform
        if self.mode == 'train' or self.mode == 'valid':
            self.labels_df, self.coordinate_df, self.series_description_df = load_train_files(relative_directory=relative_directory)
        else:
            self.labels_df = None
            self.coordinate_df = None
            self.series_description_df = load_test_files(relative_directory=relative_directory)

        logger.info(f'Loading {len(study_ids)} Studies')
        self.studies = [Study(study_id=study_id, labels_df=self.labels_df, series_description_df=self.series_description_df, coordinate_df=self.coordinate_df) for study_id in study_ids]
        logger.info(f'Done')

        self.label_columns = sum([[create_column(condition, level=level) for level in LEVELS] for condition in CONDITIONS if condition in conditions], [])
        self.series_description = series_description

        if self.mode == 'train' or self.mode == 'valid':
            series2cond = {'Sagittal T2/STIR': 'spinal',  'Sagittal T1': 'foraminal', 'Axial T2': 'subarticular'}
            self.available_diagnosis = [c for c in self.label_columns if series2cond[self.series_description] in c]
            

    @property
    def labels(self):
        return self.label_columns

    def __getitem__(self, idx):
        final_size = int(self.image_size[0]), int(self.image_size[1]), int(self.channels)
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

            centers = np.array([list(series.diagnosis_coordinates[k]) if k in series.diagnosis_coordinates else [-1.e4, -1.e4, -1] for k in self.available_diagnosis])
            # slice = int(np.random.choice([c for c in centers[:, 2] if c >=0]))
            centers = torch.as_tensor(centers[:, :2]).float()

            data = series.data
            data = data.transpose(1, 2, 0)
            H, W, D = data.shape
            if H > W:
                diff = H-W
                if self.mode == 'train':
                    offset = np.random.randint(diff)
                else:
                    offset = int(diff//2)
                data = data[offset:offset+W]
                centers[:,1] -= offset
                H = W
            elif W > H:
                diff = W-H
                if self.mode == 'train':
                    offset = np.random.randint(diff)
                else:
                    offset = int(diff//2)

                data = data[:, offset:offset+H]
                centers[:,0] -= offset
                W = H

            # logger.info(f'Data Shape : {data.shape}')
            data = cv2.resize(data, self.image_size, interpolation=cv2.INTER_LANCZOS4)

            centers[:,0] = centers[:,0]*self.image_size[0]/W
            centers[:,1] = centers[:,1]*self.image_size[1]/H

            # Select the middle portion number of channels
            i0 = int((D - self.channels) // 2)
            for i in range(self.channels):
                j = i0 + i
                x[..., i] = data[..., j]

            if self.transform is not None:
                x = self.transform(image=x)['image']

            if self.mode == 'train':
                x, centers = augment_image_and_centers(image=x, centers=centers, alpha=self.aug_size)
        else:
            centers = torch.as_tensor([[-1.e4, -1.e4] for k in self.available_diagnosis]).float()

        x = x.transpose(2, 0, 1)

        target = {}
        target['labels'] = torch.tensor(label)
        target['centers'] = centers
        target['study_id'] = torch.tensor([study.study_id])

        return torch.tensor(x, dtype=torch.float) / 255.0, target

    def __len__(self):
        return len(self.study_ids)