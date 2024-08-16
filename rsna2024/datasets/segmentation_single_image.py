import logging
import numpy as np
import random
import math
import pandas as pd


import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision

try:
    from utils import relative_directory
    import utils as rsnautils
    from datasets import load_train_files, load_test_files, LEVELS, CONDITIONS, create_column
    import datasets.dicom_load as dcmload
except ImportError:
    from ..utils import relative_directory
    from .. import utils as rsnautils
    from ..datasets import load_train_files, load_test_files, LEVELS, CONDITIONS, create_column
    from . import dicom_load as dcmload

logger = logging.getLogger(__name__)


def rotate_image(image, angle, interpolation=cv2.INTER_LANCZOS4):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=interpolation)
    return result


def augment_image_and_centers(image,centers,alpha):
#   Randomly rotate the image.
    shp = image.shape
    if len(shp) == 3:
        H, W, D = shp
    else:
        H, W = shp
        D = 1
    assert H == W
    PATCH_SIZE = H

    ## These augmentations do terribly on predicting the location of the spinal columns - It does rely on top and bottom of image
    # if random.random() > .5:
    #     image = np.flip(image, axis=1)
    #     centers[:,0] = PATCH_SIZE - centers[:,0]
    # # Randomly flip the image vertically.
    # if random.random() > 0.5:
    #     image = np.flip(image, axis=0)
    #     centers[:,1] = PATCH_SIZE - centers[:,1]


    angle = random.uniform(-180, 180)*alpha
    #TODO: We can rotate channel by channel to use the better interpolation
    image = rotate_image(image, angle, interpolation=cv2.INTER_LANCZOS4 if D <= 3 else cv2.INTER_LINEAR)
    if D == 1 and len(image.shape) == 2:
        image = np.expand_dims(image, 2)
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
            self.labels_df, self.coordinate_df, self.series_description_df = load_train_files(relative_directory=relative_directory, clean=rsnautils.CLEAN)
        else:
            self.labels_df = None
            self.coordinate_df = None
            self.series_description_df = load_test_files(relative_directory=relative_directory)

        # Check all study ids are in loaded files
        
        assert len(set(study_ids) & set(self.series_description_df.study_id.unique())) == len(study_ids)

        logger.info(f'Loading {len(study_ids)} Studies')
        self.studies = [dcmload.OrientedStudy(study_id=study_id, labels_df=self.labels_df, series_description_df=self.series_description_df, coordinate_df=self.coordinate_df) for study_id in study_ids]
        logger.info(f'Done')

        self.label_columns = sum([[create_column(condition, level=level) for level in LEVELS] for condition in CONDITIONS if condition in conditions], [])
        self.series_description = series_description

        if self.mode == 'train' or self.mode == 'valid':
            series2cond = {'Sagittal T2/STIR': 'spinal',  'Sagittal T1': 'foraminal', 'Axial T2': 'subarticular'}
            self.available_diagnosis = [c for c in self.label_columns if series2cond[self.series_description] in c]
            
    @property
    def labels(self):
        return self.label_columns


    def get_instance_number(self, series):
        stack = series.get_largest_stack()
        instance_number = stack.instance_numbers[stack.number_of_instances//2]
        return instance_number


    def __getitem__(self, idx):
        
        study = self.studies[idx]

        full_targets = self.mode == 'train' or self.mode == 'valid'
        target = {'study_id': torch.tensor([study.study_id])}
        if full_targets:
            label = np.int64([study.labels[col] for col in self.label_columns])
            target['labels'] = torch.tensor(label)
        
        series = study.get_largest_series(self.series_description)
        target['series_id'] = torch.tensor([series.series_id]) if series is not None else torch.tensor([-1])
        if series is not None:
            instance_number = self.get_instance_number(series=series)
            stack = series.get_largest_stack()
            data, instance_numbers = stack.get_thick_slice(instance_number=instance_number, slice_thickness=self.channels)
            target['instance_numbers'] = torch.as_tensor(instance_numbers, dtype=torch.long)
            offsets = np.zeros(2)
            scalings = np.ones(2)

            data = data.transpose(1, 2, 0)
            H, W, D = data.shape
            if H > W:
                diff = H-W
                if self.mode == 'train':
                    offset = np.random.randint(diff)
                else:
                    offset = int(diff//2)
                data = data[offset:offset+W]
                offsets[0] = -offset
                H = W
            elif W > H:
                diff = W-H
                if self.mode == 'train':
                    offset = np.random.randint(diff)
                else:
                    offset = int(diff//2)

                data = data[:, offset:offset+H]
                offsets[1] = -offset
                W = H

            data = cv2.resize(data, self.image_size, interpolation=cv2.INTER_LANCZOS4)
            if len(data.shape) == 2:
                data = np.expand_dims(data, 2)
            scalings[0] = self.image_size[0]/H
            scalings[1] = self.image_size[1]/W

            target['offsets'] = torch.tensor(offsets)
            target['scalings'] = torch.tensor(scalings)

            if full_targets:
                series_mask = self.coordinate_df.series_id == series.series_id
                used_instances = self.coordinate_df.loc[series_mask, 'instance_number'].unique()
                slice_classification = np.array([(1 if j in used_instances else 0) for j in instance_numbers], dtype=int)
                target['slice_classification'] = torch.as_tensor(slice_classification).long()

                tmp = self.coordinate_df.loc[series_mask].set_index('level')
                
                centers = np.array([tmp.loc[level, ['x', 'y']].values if level in tmp.index else np.array([-1.e4, -1.e4]) for level in LEVELS], dtype=float)

                centers += offsets
                centers *= scalings
                centers = torch.tensor(centers, dtype=torch.float)
                target['centers'] = centers

            if self.transform is not None:
                data = self.transform(image=data)['image']

            if self.mode == 'train':
                data, centers = augment_image_and_centers(image=data, centers=centers, alpha=self.aug_size)
                target['centers'] = centers
        else:
            final_size = int(self.image_size[0]), int(self.image_size[1]), int(self.channels)
            data = np.zeros(final_size)
            if full_targets:
                target['centers'] = torch.ones((5,2), dtype=torch.float) * -1.e4
                target['offsets'] = torch.zeros((2,), dtype=torch.float)
                target['scalings'] = torch.ones((2,), dtype=torch.float)
                target['slice_classification'] = torch.zeros((self.channels, ), dtype=torch.long)
                target['instance_numbers'] = torch.ones((self.channels, ), dtype=torch.long)*-1

        data = data.transpose(2, 0, 1)

        return torch.tensor(data, dtype=torch.float) / 255.0, target

    def __len__(self):
        return len(self.study_ids)
    


class SegmentationPredictedCenterDataset(SegmentationCenterDataset):
    def __init__(self, study_ids, image_size, channels: int, series_description, conditions, generated_coordinate_file, mode='train', aug_size=0.0, transform=None):
        self.pred_center_df = pd.read_csv(generated_coordinate_file)  # Should be a file just like any other coordinate file
        super().__init__(study_ids=study_ids, image_size=image_size, channels=channels, series_description=series_description, conditions=conditions, mode=mode, aug_size=aug_size, transform=transform)
        
    def get_instance_number(self, series):
        i = self.pred_center_df.loc[self.pred_center_df.series_id == series.series_id, 'instance_number'].values
        median = int(np.median(i))
        if median not in i:
            return sorted(i, key=lambda v: abs(v-i))[0]

        return median