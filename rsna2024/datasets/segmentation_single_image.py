import logging
import numpy as np
import random
import math
import pandas as pd
import copy


import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision

try:
    import utils as rsnautils
    from datasets import load_train_files, load_test_files, LEVELS, CONDITIONS, create_column
    import datasets.dicom_load as dcmload
except ImportError:
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
            self.labels_df, self.coordinate_df, self.series_description_df = load_train_files(relative_directory=rsnautils.relative_directory, clean=rsnautils.CLEAN)
        else:
            self.labels_df = None
            self.coordinate_df = None
            logger.info(f'In {self.mode} Mode for {study_ids}')
            self.series_description_df = load_test_files(relative_directory=rsnautils.relative_directory)

        # Check all study ids are in loaded files
        
        #assert len(set(study_ids) & set(self.series_description_df.study_id.unique())) == len(study_ids), f"for mode {self.mode}, clean = {rsnautils.CLEAN}, unable to find all {study_ids}"

        logger.info(f'Loading {len(study_ids)} Studies')
        self.studies = [dcmload.OrientedStudy(study_id=study_id, labels_df=self.labels_df, series_description_df=self.series_description_df, coordinate_df=self.coordinate_df) for study_id in study_ids]
        logger.info(f'Done')

        self.label_columns = sum([[create_column(condition, level=level) for level in LEVELS] for condition in CONDITIONS if condition in conditions], [])
        self.series_description = series_description

        self.fails = set()

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
        target = {'study_id': study.study_id}
        data = {'study_id': study.study_id}
        if full_targets:
            label = np.int64([study.labels[col] for col in self.label_columns])
            target['labels'] = torch.tensor(label)
        try:       
            study.load() 
            series = study.get_largest_series(self.series_description)
            data['series_id'] = series.series_id if series is not None else -1
            if series is not None:

                instance_number = self.get_instance_number(series=series)
                stack = series.get_largest_stack()
                img, instance_numbers = stack.get_thick_slice(instance_number=instance_number, slice_thickness=self.channels)
                data['instance_numbers'] = torch.as_tensor(instance_numbers, dtype=torch.long)
                offsets = np.zeros(2)
                scalings = np.ones(2)

                img = img.transpose(1, 2, 0)
                H, W, D = img.shape
                # For training, we do some shrinking or growing as is necessary to get a full image
                # In this case, we will put the image in the center of a larger patch with zero padding
                diff = H-W
                S = max(H,W)
                new_img = np.zeros((S,S,D), dtype=np.uint8)
                if diff > 0:  #Height is greater than width
                    d = diff // 2
                    new_img[:, d:d+W, :] = img
                    offsets[1] = d
                else:
                    d = -diff//2
                    new_img[d:d+H, :, :] = img
                    offsets[0] = d

                img = new_img
                H = W = S

                if self.mode == 'train':
                    # Do a random crop augmentation
                    sc = np.random.uniform()*self.aug_size
                    S2 = int(round(S*(1-sc)))
                    dS = S-S2
                    if dS > 0:
                        dh = np.random.randint(dS)
                        dw = np.random.randint(dS)
                    else:
                        dh = 0
                        dw = 0
                    img = img[dh:dh+S2, dw:dw+S2]
                    offsets[0] -= dh
                    offsets[1] -= dw
                    H = W = S2

                img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LANCZOS4)
                if len(img.shape) == 2:
                    img = np.expand_dims(img, 2)
                scalings[0] = self.image_size[0]/H
                scalings[1] = self.image_size[1]/W

                data['offsets'] = torch.tensor(offsets)
                data['scalings'] = torch.tensor(scalings)

                if full_targets:
                    series_mask = self.coordinate_df.series_id == series.series_id
                    used_instances = self.coordinate_df.loc[series_mask, 'instance_number'].unique()
                    slice_classification = np.array([(1 if j in used_instances else 0) for j in instance_numbers], dtype=int)
                    target['slice_classification'] = torch.as_tensor(slice_classification).long()

                    level_slice_classification = np.zeros((5, self.channels), dtype=int)
                    for k, lev in enumerate(LEVELS):
                        used_instances = self.coordinate_df.loc[series_mask & (self.coordinate_df.level == lev), 'instance_number'].unique()
                        level_slice_classification[k] = np.array([(1 if j in used_instances else 0) for j in instance_numbers], dtype=int)
                    target['level_slice_classification'] = torch.as_tensor(level_slice_classification).long()

                    tmp = self.coordinate_df.loc[series_mask].set_index('level')
                    
                    centers = np.array([tmp.loc[level, ['x', 'y']].values if level in tmp.index else np.array([-1.e4, -1.e4]) for level in LEVELS], dtype=float)

                    centers += offsets
                    centers *= scalings
                    centers = torch.tensor(centers, dtype=torch.float)
                    target['centers'] = centers

                if self.transform is not None:
                    img = self.transform(image=img)['image']

                if self.mode == 'train':
                    img, centers = augment_image_and_centers(image=img, centers=centers, alpha=self.aug_size)
                    target['centers'] = centers

            else:
                for j in range(5):
                    self.fails.add((study.study_id, self.series_description, j))
                logger.error(f'No {self.series_description} series for {study.study_id}')
                final_size = int(self.image_size[0]), int(self.image_size[1]), int(self.channels)
                img = np.zeros(final_size)
                data['series_id'] = -1
                data['instance_numbers'] = torch.ones((self.channels, ), dtype=torch.long)*-1
                data['offsets'] = torch.zeros((2,), dtype=torch.float)
                data['scalings'] = torch.ones((2,), dtype=torch.float)
                if full_targets:
                    target['centers'] = torch.ones((5,2), dtype=torch.float) * -1.e4    
                    target['slice_classification'] = torch.zeros((self.channels, ), dtype=torch.long)
                    target['level_slice_classification'] = torch.zeros((5, self.channels), dtype=torch.long)
        except Exception as e:
            logger.exception(e)
            logger.error(f'{study.study_id}')
            for j in range(5):
                self.fails.add((study.study_id, self.series_description, j))
            final_size = int(self.image_size[0]), int(self.image_size[1]), int(self.channels)
            img = np.zeros(final_size)
            data['series_id'] = -1
            data['instance_numbers'] = torch.ones((self.channels, ), dtype=torch.long)*-1
            data['offsets'] = torch.zeros((2,), dtype=torch.float)
            data['scalings'] = torch.ones((2,), dtype=torch.float)
            if full_targets:
                target['centers'] = torch.ones((5,2), dtype=torch.float) * -1.e4    
                target['slice_classification'] = torch.zeros((self.channels, ), dtype=torch.long)                    
                target['level_slice_classification'] = torch.zeros((5, self.channels), dtype=torch.long)

        img = img.transpose(2, 0, 1)

        data[f'{self.series_description} Patch'] = torch.tensor(img, dtype=torch.float) / 255.0

        if self.mode == 'test':
            study.unload()

        return copy.deepcopy(data), copy.deepcopy(target)

    def __len__(self):
        return len(self.study_ids)
    


class SegmentationPredictedCenterDataset(SegmentationCenterDataset):
    def __init__(self, study_ids, image_size, channels: int, series_description, conditions, generated_coordinate_file, mode='train', aug_size=0.0, transform=None):
        self.pred_center_df = pd.read_csv(generated_coordinate_file)  # Should be a file just like any other coordinate file
        super().__init__(study_ids=study_ids, image_size=image_size, channels=channels, series_description=series_description, conditions=conditions, mode=mode, aug_size=aug_size, transform=transform)
        
    def get_instance_number(self, series):
        try:
            ### This actually a problem, since the instance numbers are not necessarily ordered
            i = self.pred_center_df.loc[self.pred_center_df.series_id == series.series_id, 'instance_number'].values
            median = int(np.median(i))
            if median not in i:
                return sorted(i, key=lambda v: abs(v-i))[0]

            return median
        except Exception:
            return super().get_instance_number(series)