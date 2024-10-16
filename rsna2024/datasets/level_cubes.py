import logging
import numpy as np
import pandas as pd
import math

import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision

try:
    import utils as rsnautils
    from datasets import load_train_files, load_test_files, LEVELS, CONDITIONS, create_column
    from datasets.dicom_load import OrientedStudy
except ImportError:
    from .. import utils as rsnautils
    from ..datasets import load_train_files, load_test_files, LEVELS, CONDITIONS, create_column
    from .dicom_load import OrientedStudy

logger = logging.getLogger(__name__)



class LevelCubeDataset(Dataset):
    def __init__(self, study_ids, channels: int, patch_size: int, series_description: str, condition: str, generated_coordinate_file: str, mode='train', transform: callable = None, load_studies: list[OrientedStudy]|None = None):
        self.study_ids = list(study_ids)
        self.patch_size = int(patch_size)
        self.channels = int(channels)
        logger.info(f'Output will have patch_size {self.patch_size} and {channels} channels')
        self.mode = mode
        self.transform = transform
        if self.mode == 'train' or self.mode == 'valid':
            self.labels_df, self.coordinate_df, self.series_description_df = load_train_files(relative_directory=rsnautils.relative_directory, clean=rsnautils.CLEAN)
        else:
            self.labels_df = None
            self.coordinate_df = None
            self.series_description_df = load_test_files(relative_directory=rsnautils.relative_directory)

        if load_studies is None:
            logger.info(f'Loading {len(study_ids)} Studies')
            self.studies = [OrientedStudy(study_id=study_id, labels_df=self.labels_df, series_description_df=self.series_description_df, coordinate_df=self.coordinate_df) for study_id in study_ids]
            logger.info(f'Done')
        else:
            logger.info(f'Referencing pre-loaded studies')
            self.studies = load_studies
            assert len(self.study_ids) == len(self.studies)


        self.label_columns = [create_column(condition, level=level) for level in LEVELS]
        self.series_description = series_description
        self.condition = condition
            
        self.pred_center_df = pd.read_csv(generated_coordinate_file)  # Should be a file just like any other coordinate file

    @property
    def labels(self):
        return self.label_columns

    def __getitem__(self, idx):
        """
        Will output a tensor of size LEVELS, CHANNELS, PATCH_SIZE, PATCH_SIZE
        """
        final_size = 5, self.patch_size, self.patch_size, int(self.channels)
        x = np.zeros(final_size, dtype=np.uint8)
        study = self.studies[idx]
        study.load()
        full_targets = self.mode == 'train' or self.mode == 'valid'
        target = {'study_id': torch.tensor([study.study_id])}
        if full_targets:
            label = np.int64([study.labels[col] for col in self.label_columns])
            target['labels'] = torch.tensor(label)

        # Find all points from coordinate dataframe for this study
        tmp = self.pred_center_df[(self.pred_center_df.study_id == study.study_id) & (self.pred_center_df.condition == self.condition)].sort_values('level')
        assert len(tmp) <= 5
        # For each level, find the series and extract the patch
        saves = []
        level_dict = {lev: i for i, lev in enumerate(LEVELS)}
        patch_offsets = np.zeros((5, 2), dtype=int)
        series_ids = np.zeros((5,), dtype=int)
        for row in tmp.itertuples():
            try:
                series = study.get_series(row.series_id)
                x0=int(round(row.x)) 
                y0=int(round(row.y))
                if self.series_description == 'Axial T2':
                    x0, y0 = y0, x0

                inum = row.instance_number
                if self.mode == 'train':
                    gap = int(self.patch_size // 10)
                    x0 += np.random.randint(-gap, gap+1)
                    y0 += np.random.randint(-gap, gap+1)
                    stack = series.get_stack(row.instance_number)
                    k = stack._get_instance_k(row.instance_number)
                    k = np.clip(k + np.random.randint(-1, 2), 0, stack.number_of_instances-1)
                    inum = stack.instance_numbers[k]


                patch, instance_numbers, patch_offset = series.get_thick_patch(instance_number=inum, slice_thickness=self.channels, x=x0, y=y0, patch_size=self.patch_size)
                lev = level_dict[row.level]
                x[lev] = patch.transpose(1,2,0)
                patch_offsets[lev, 0] = patch_offset[0]
                patch_offsets[lev, 1] = patch_offset[1]
                series_ids[lev] = row.series_id
                saves.append((lev, row.level, series.series_id, instance_numbers, patch_offset))
            except (KeyError, ValueError):
                pass

        target['patch_offsets'] = torch.tensor(patch_offsets, dtype=torch.float)
        target['series_ids'] = torch.tensor(series_ids, dtype=torch.long)
        if full_targets:
            study_mask = (self.coordinate_df.study_id == study.study_id) & (self.coordinate_df.condition == self.condition)
            tmp2 = self.coordinate_df.loc[study_mask]
            centers = np.zeros((5, 2), dtype=float) - 1000
            slice_classification = np.zeros((5,self.channels), dtype=int)
            for lev, level, series_id, instance_numbers, patch_offset in saves:
                series = study.get_series(series_id)
                used_instances = tmp2.loc[(tmp2.series_id == series_id) & (tmp2.level == level), 'instance_number'].unique()
                
                slice_classification[lev] = np.array([(1 if j in used_instances else 0) for j in instance_numbers], dtype=int)
                tmp3 = tmp2.loc[(tmp2.series_id == series_id) & (tmp2.level == level), ['x', 'y']]
                if len(tmp3) == 1:
                    x0, y0 = tmp3.iloc[0].values
                    x0 -= patch_offset[0]
                    y0 -= patch_offset[1]
                    centers[lev] = x0, y0
            target['centers'] = torch.tensor(centers, dtype=torch.float)
            target['slice_classification'] = torch.as_tensor(slice_classification).long()

                    
        if self.transform is not None:
            # Need to reshape it around
            x = x.transpose(1, 2, 3, 0).reshape(self.patch_size, self.patch_size, -1)
            x = self.transform(image=x)['image'].reshape(self.patch_size, self.patch_size, self.channels, -1).transpose(3, 0, 1, 2)
    
        x = x.transpose(0, 3, 1, 2)
        if self.mode == 'test':
            study.unload()
        return torch.tensor(x, dtype=torch.float) / 255.0, target

    def __len__(self):
        return len(self.study_ids)
    
### TODO: Fix this part also so we extract everything for a single condition from all three types of images
class AllLevelCubeDataset(LevelCubeDataset):
    def __init__(self, study_ids, series_description: str, condition: str, 
                 sagittal_t2_channels: int, sagittal_t2_patch_size: int,
                 sagittal_t1_channels: int, sagittal_t1_patch_size: int,
                 axial_t2_channels: int, axial_t2_patch_size: int,
                 generated_coordinate_file: str, mode='train', transform: callable = None, load_studies: list[OrientedStudy]|None = None):
        self.study_ids = list(study_ids)
        self.sagittal_t2_patch_size = int(sagittal_t2_patch_size)
        self.sagittal_t2_channels = int(sagittal_t2_channels)
        self.sagittal_t1_patch_size = int(sagittal_t1_patch_size)
        self.sagittal_t1_channels = int(sagittal_t1_channels)
        self.axial_t2_patch_size = int(axial_t2_patch_size)
        self.axial_t2_channels = int(axial_t2_channels)
        # logger.info(f'Output will have patch_size {self.patch_size} and {channels} channels')
        self.mode = mode
        self.transform = transform
        if self.mode == 'train' or self.mode == 'valid':
            self.labels_df, self.coordinate_df, self.series_description_df = load_train_files(relative_directory=rsnautils.relative_directory, clean=rsnautils.CLEAN)
        else:
            self.labels_df = None
            self.coordinate_df = None
            self.series_description_df = load_test_files(relative_directory=rsnautils.relative_directory)

        if load_studies is None:
            logger.info(f'Loading {len(study_ids)} Studies')
            self.studies = [OrientedStudy(study_id=study_id, labels_df=self.labels_df, series_description_df=self.series_description_df, coordinate_df=self.coordinate_df) for study_id in study_ids]
            logger.info(f'Done')
        else:
            logger.info(f'Referencing pre-loaded studies')
            self.studies = load_studies
            assert len(self.study_ids) == len(self.studies)


        self.label_columns = [create_column(condition, level=level) for level in LEVELS]
        self.series_description = series_description
        self.condition = condition
            
        self.pred_center_df = pd.read_csv(generated_coordinate_file)  # Should be a file just like any other coordinate file

    @property
    def labels(self):
        return self.label_columns

    def get_cube(self, study, series_description, world_x, world_y, world_z, patch_size, channels):
        try:
            target_stack, target_series, dist = study.find_closest_stack_and_series(series_description=series_description, world_x=world_x, world_y=world_y, world_z=world_z, required_in=False)
            if target_stack is not None:
                k, proj_x, proj_y = target_stack.get_pixel_from_world(world_x=world_x, world_y=world_y, world_z=world_z)
                k = np.clip(k, 0, len(target_stack.instance_numbers) - 1)
                inum, px, py = target_stack.instance_numbers[k], proj_x, proj_y
                if series_description == 'Axial T2':
                    px, py = py, px
                patch, instance_numbers, patch_offset = target_stack.get_thick_patch(instance_number=inum, slice_thickness=channels, x=px, y=py, patch_size=patch_size)
                return patch
        except np.linalg.LinAlgError:
            pass
        final_size = channels, patch_size, patch_size
        return np.zeros(final_size, dtype=np.uint8)

    def __getitem__(self, idx):
        """
        Will output a tensor of size LEVELS, CHANNELS, PATCH_SIZE, PATCH_SIZE
        """
        
        study = self.studies[idx]
        study.load()
        full_targets = self.mode == 'train' or self.mode == 'valid'
        target = {'study_id': torch.tensor([study.study_id])}
        if full_targets:
            label = np.int64([study.labels[col] for col in self.label_columns])
            target['labels'] = torch.tensor(label)

        # Find all points from coordinate dataframe for this study
        tmp = self.pred_center_df[(self.pred_center_df.study_id == study.study_id) & (self.pred_center_df.condition == self.condition)].sort_values('level')
        assert len(tmp) <= 5
        # For each level, find the series and extract the patch
        level_dict = {lev: i for i, lev in enumerate(LEVELS)}
        world_coordinates = np.zeros((5, 3), dtype=float)
        for row in tmp.itertuples():
                try:
                    series = study.get_series(row.series_id)
                    stack = series.get_stack(row.instance_number)
                    lev = level_dict[row.level]
                    world_coordinates[lev] = stack.get_world_coordinates(instance_number=row.instance_number, x=row.x, y=row.y)
                except KeyError:
                    pass

        sagittal_t2_x = np.zeros((5, self.sagittal_t2_channels, self.sagittal_t2_patch_size, self.sagittal_t2_patch_size), dtype=np.uint8)
        sagittal_t1_x = np.zeros((5, self.sagittal_t1_channels, self.sagittal_t1_patch_size, self.sagittal_t1_patch_size), dtype=np.uint8)
        axial_t2_x = np.zeros((5, self.axial_t2_channels, self.axial_t2_patch_size, self.axial_t2_patch_size), dtype=np.uint8)

        for i in range(5):
            world_x, world_y, world_z = world_coordinates[i]
            # TODO: Add a small randomness here
            sagittal_t2_x[i] = self.get_cube(study, 'Sagittal T2/STIR', world_x, world_y, world_z, self.sagittal_t2_patch_size, self.sagittal_t2_channels)
            sagittal_t1_x[i] = self.get_cube(study, 'Sagittal T1', world_x, world_y, world_z, self.sagittal_t1_patch_size, self.sagittal_t1_channels)
            axial_t2_x[i] = self.get_cube(study, 'Axial T2', world_x, world_y, world_z, self.axial_t2_patch_size, self.axial_t2_channels)

        if self.transform is not None:
            # Need to reshape it around

            sagittal_t2_x = sagittal_t2_x.transpose(2, 3, 1, 0).reshape(self.sagittal_t2_patch_size, self.sagittal_t2_patch_size, -1)
            sagittal_t2_x = self.transform(image=sagittal_t2_x)['image'].reshape(self.sagittal_t2_patch_size, self.sagittal_t2_patch_size, self.sagittal_t2_channels, -1).transpose(3, 2, 0, 1)

            sagittal_t1_x = sagittal_t1_x.transpose(2, 3, 1, 0).reshape(self.sagittal_t1_patch_size, self.sagittal_t1_patch_size, -1)
            sagittal_t1_x = self.transform(image=sagittal_t1_x)['image'].reshape(self.sagittal_t1_patch_size, self.sagittal_t1_patch_size, self.sagittal_t1_channels, -1).transpose(3, 2, 0, 1)

            axial_t2_x = axial_t2_x.transpose(2, 3, 1, 0).reshape(self.axial_t2_patch_size, self.axial_t2_patch_size, -1)
            axial_t2_x = self.transform(image=axial_t2_x)['image'].reshape(self.axial_t2_patch_size, self.axial_t2_patch_size, self.axial_t2_channels, -1).transpose(3, 2, 0, 1)
    
        if self.mode == 'test':
            study.unload()

        return (torch.tensor(sagittal_t2_x, dtype=torch.float) / 255.0, torch.tensor(sagittal_t1_x, dtype=torch.float) / 255.0, torch.tensor(axial_t2_x, dtype=torch.float) / 255.0), target

    def __len__(self):
        return len(self.study_ids)

   


class LevelCubeLeftRightDataset(Dataset):
    def __init__(self, study_ids, channels: int, patch_size: int, series_description: str, left_condition: str, right_condition: str, generated_coordinate_file: str, mode='train', transform: callable = None, load_studies: list[OrientedStudy]|None = None):
        self.study_ids = list(study_ids)
        self.patch_size = int(patch_size)
        self.channels = channels
        logger.info(f'Output will have size {self.patch_size} and {channels} channels')
        self.mode = mode
        self.transform = transform
        
        self.label_columns = sum([[create_column(condition, level=level) for level in LEVELS] for condition in CONDITIONS if condition in [left_condition, right_condition]], [])
        assert len(self.label_columns) == 10
        self.series_description = series_description
        self.left_dataset = LevelCubeDataset(study_ids=study_ids, 
                                             channels=channels, 
                                             patch_size=patch_size, 
                                             series_description=series_description, 
                                             condition=left_condition, 
                                             generated_coordinate_file=generated_coordinate_file, 
                                             mode=mode, 
                                             transform=transform, 
                                             load_studies=None)
        self.right_dataset = LevelCubeDataset(study_ids=study_ids, 
                                             channels=channels, 
                                             patch_size=patch_size, 
                                             series_description=series_description, 
                                             condition=right_condition, 
                                             generated_coordinate_file=generated_coordinate_file, 
                                             mode=mode, 
                                             transform=transform, 
                                             load_studies=self.left_dataset.studies)
            
    @property
    def labels(self):
        return self.label_columns

    def __getitem__(self, idx):
        """
        Will output a tensor of size LEVELS, SIDE, CHANNELS, 2*SLICE_SIZE, 2*SLICE_SIZE
        """
        study = self.left_dataset.studies[idx]
        full_targets = self.mode == 'train' or self.mode == 'valid'
        target = {'study_id': torch.tensor([study.study_id])}
        if full_targets:
            label = np.int64([study.labels[col] for col in self.label_columns])
            target['labels'] = torch.tensor(label)
        left_x, left_targets = self.left_dataset[idx]
        right_x, right_targets = self.right_dataset[idx]
        if self.series_description == 'Axial T2':
            x = torch.stack([left_x, torch.flip(right_x, dims=[-1])], dim=1)
        else:
            x = torch.stack([left_x, right_x], dim=1)

        for key in left_targets.keys():
            if key not in ['study_id', 'labels']:
                left_t = left_targets[key]
                right_t = right_targets[key]
                target[key] = torch.stack([left_t, right_t], dim=1)

        return x, target
        
    def __len__(self):
        return len(self.left_dataset)


class AllLevelCubeLeftRightDataset(Dataset):
    def __init__(self, study_ids, 
                 sagittal_t2_channels: int, sagittal_t2_patch_size: int,
                 sagittal_t1_channels: int, sagittal_t1_patch_size: int,
                 axial_t2_channels: int, axial_t2_patch_size: int,
                 series_description: str, 
                 left_condition: str, 
                 right_condition: str, 
                 generated_coordinate_file: str, mode='train', transform: callable = None, load_studies: list[OrientedStudy]|None = None):
        self.study_ids = list(study_ids)
        self.mode = mode
        self.transform = transform
        
        self.label_columns = sum([[create_column(condition, level=level) for level in LEVELS] for condition in CONDITIONS if condition in [left_condition, right_condition]], [])
        assert len(self.label_columns) == 10
        self.series_description = series_description
        self.left_dataset = AllLevelCubeDataset(study_ids=study_ids, 
                                             sagittal_t2_channels=sagittal_t2_channels, sagittal_t2_patch_size=sagittal_t2_patch_size,
                                             sagittal_t1_channels=sagittal_t1_channels, sagittal_t1_patch_size=sagittal_t1_patch_size,
                                             axial_t2_channels=axial_t2_channels, axial_t2_patch_size=axial_t2_patch_size,
                                             series_description=series_description, 
                                             condition=left_condition, 
                                             generated_coordinate_file=generated_coordinate_file, 
                                             mode=mode, 
                                             transform=transform, 
                                             load_studies=None)
        self.right_dataset = AllLevelCubeDataset(study_ids=study_ids, 
                                             sagittal_t2_channels=sagittal_t2_channels, sagittal_t2_patch_size=sagittal_t2_patch_size,
                                             sagittal_t1_channels=sagittal_t1_channels, sagittal_t1_patch_size=sagittal_t1_patch_size,
                                             axial_t2_channels=axial_t2_channels, axial_t2_patch_size=axial_t2_patch_size,
                                             series_description=series_description, 
                                             condition=right_condition, 
                                             generated_coordinate_file=generated_coordinate_file, 
                                             mode=mode, 
                                             transform=transform, 
                                             load_studies=self.left_dataset.studies)
            
    @property
    def labels(self):
        return self.label_columns

    def __getitem__(self, idx):
        """
        Will output a tensor of size LEVELS, SIDE, CHANNELS, 2*SLICE_SIZE, 2*SLICE_SIZE
        """
        study = self.left_dataset.studies[idx]
        full_targets = self.mode == 'train' or self.mode == 'valid'
        target = {'study_id': torch.tensor([study.study_id])}
        if full_targets:
            label = np.int64([study.labels[col] for col in self.label_columns])
            target['labels'] = torch.tensor(label)
        left_sagittal_t2_x, left_sagittal_t1_x, left_axial_t2_x, left_targets = self.left_dataset[idx]
        right_sagittal_t2_x, right_sagittal_t1_x, right_axial_t2_x, right_targets = self.right_dataset[idx]

        if self.series_description == 'Axial T2':
            sagittal_t2_x = torch.stack([left_sagittal_t2_x, torch.flip(right_sagittal_t2_x, dim=[-3])], dim=1)
            sagittal_t1_x = torch.stack([left_sagittal_t1_x, torch.flip(right_sagittal_t1_x, dim=[-3])], dim=1)
            axial_t2_x = torch.stack([left_axial_t2_x, torch.flip(right_axial_t2_x, dim=[-1])], dim=1)
        else:
            sagittal_t2_x = torch.stack([left_sagittal_t2_x, right_sagittal_t2_x], dim=1)
            sagittal_t1_x = torch.stack([left_sagittal_t1_x, right_sagittal_t1_x], dim=1)
            axial_t2_x = torch.stack([left_axial_t2_x, right_axial_t2_x], dim=1)

        for key in left_targets.keys():
            if key not in ['study_id', 'labels']:
                left_t = left_targets[key]
                right_t = right_targets[key]
                target[key] = torch.stack([left_t, right_t], dim=1)

        return (sagittal_t2_x, sagittal_t1_x, axial_t2_x), target
        
    def __len__(self):
        return len(self.left_dataset)


class LevelCubeAreaDataset(Dataset):
    def __init__(self, study_ids, channels: int, patch_size: int, d_side: float, series_description: str, condition: str, generated_coordinate_file: str, mode='train', transform: callable = None, load_studies: list[OrientedStudy]|None = None):
        self.study_ids = list(study_ids)
        self.patch_size = int(patch_size)
        self.channels = int(channels)
        self.d_side = d_side
        logger.info(f'Output will have patch_size {self.patch_size} and {channels} channels, cover area with side of {d_side} mm')
        self.mode = mode
        self.transform = transform
        if self.mode == 'train' or self.mode == 'valid':
            self.labels_df, self.coordinate_df, self.series_description_df = load_train_files(relative_directory=rsnautils.relative_directory, clean=rsnautils.CLEAN)
        else:
            self.labels_df = None
            self.coordinate_df = None
            self.series_description_df = load_test_files(relative_directory=rsnautils.relative_directory)

        if load_studies is None:
            logger.info(f'Loading {len(study_ids)} Studies')
            self.studies = [OrientedStudy(study_id=study_id, labels_df=self.labels_df, series_description_df=self.series_description_df, coordinate_df=self.coordinate_df) for study_id in study_ids]
            logger.info(f'Done')
        else:
            logger.info(f'Referencing pre-loaded studies')
            self.studies = load_studies
            assert len(self.study_ids) == len(self.studies)


        self.label_columns = [create_column(condition, level=level) for level in LEVELS]
        self.series_description = series_description
        self.condition = condition
            
        self.pred_center_df = pd.read_csv(generated_coordinate_file)  # Should be a file just like any other coordinate file

        self.fails = set()

    @property
    def labels(self):
        return self.label_columns

    def __getitem__(self, idx):
        """
        Will output a tensor of size LEVELS, CHANNELS, PATCH_SIZE, PATCH_SIZE
        """
        final_size = 5, self.patch_size, self.patch_size, int(self.channels)
        x = np.zeros(final_size, dtype=np.uint8)
        full_targets = self.mode == 'train' or self.mode == 'valid'
        study = self.studies[idx]
        target = {'study_id': study.study_id}
        patch_offsets = np.zeros((5, 2), dtype=int)
        patch_scalings = np.zeros((5, 2), dtype=int)
        series_ids = np.zeros((5,), dtype=int)

        try:
            study.load()
            if full_targets:
                label = np.int64([study.labels[col] for col in self.label_columns])
                target['labels'] = torch.tensor(label)

            # Find all points from coordinate dataframe for this study
            tmp = self.pred_center_df[(self.pred_center_df.study_id == study.study_id) & (self.pred_center_df.condition == self.condition)].sort_values('level')
            # assert len(tmp) <= 5
            # For each level, find the series and extract the patch
            saves = []
            level_dict = {lev: i for i, lev in enumerate(LEVELS)}
            patch_offsets = np.zeros((5, 2), dtype=int)
            patch_scalings = np.zeros((5, 2), dtype=int)
            series_ids = np.zeros((5,), dtype=int)
            for row in tmp.itertuples():
                try:
                    series = study.get_series(row.series_id)
                    x0=int(round(row.x)) 
                    y0=int(round(row.y))
                    if self.series_description == 'Axial T2':
                        x0, y0 = y0, x0

                    inum = row.instance_number
                    if self.mode == 'train':
                        gap = int(self.patch_size // 10)
                        x0 += np.random.randint(-gap, gap+1)
                        y0 += np.random.randint(-gap, gap+1)
                        stack = series.get_stack(row.instance_number)
                        k = stack._get_instance_k(row.instance_number)
                        k = np.clip(k + np.random.randint(-1, 2), 0, stack.number_of_instances-1)
                        inum = stack.instance_numbers[k]

                    stack = series.get_stack(instance_number=inum)
                    patch, instance_numbers, patch_offset, scaling = stack.get_thick_area(instance_number=inum, slice_thickness=self.channels, x=x0, y=y0, patch_size=self.patch_size, d_mm=self.d_side)
                    lev = level_dict[row.level]
                    x[lev] = patch.transpose(1,2,0)
                    patch_offsets[lev, 0] = patch_offset[0]
                    patch_offsets[lev, 1] = patch_offset[1]
                    patch_scalings[lev, 0] = scaling[0]
                    patch_scalings[lev, 1] = scaling[1]
                    series_ids[lev] = row.series_id if isinstance(row.series_id, int) else 0
                    saves.append((lev, row.level, series.series_id, instance_numbers, patch_offset, scaling))
                except (KeyError, ValueError):
                    self.fails.append((study.study_id, self.series_description, level_dict[row.level]))
                    pass

            target['patch_offsets'] = torch.tensor(patch_offsets, dtype=torch.float)
            target['patch_scalings'] = torch.tensor(patch_scalings, dtype=torch.float)
            target['series_ids'] = torch.tensor(series_ids, dtype=torch.long)
            if full_targets:
                study_mask = (self.coordinate_df.study_id == study.study_id) & (self.coordinate_df.condition == self.condition)
                tmp2 = self.coordinate_df.loc[study_mask]
                centers = np.zeros((5, 2), dtype=float) - 1000
                slice_classification = np.zeros((5,self.channels), dtype=int)
                for lev, level, series_id, instance_numbers, patch_offset, scaling in saves:
                    series = study.get_series(series_id)
                    used_instances = tmp2.loc[(tmp2.series_id == series_id) & (tmp2.level == level), 'instance_number'].unique()
                    
                    slice_classification[lev] = np.array([(1 if j in used_instances else 0) for j in instance_numbers], dtype=int)
                    tmp3 = tmp2.loc[(tmp2.series_id == series_id) & (tmp2.level == level), ['x', 'y']]
                    if len(tmp3) == 1:
                        x0, y0 = tmp3.iloc[0].values
                        x0 /= scaling[0]
                        y0 /= scaling[1]
                        x0 -= patch_offset[0]
                        y0 -= patch_offset[1]
                        centers[lev] = x0, y0
                target['centers'] = torch.tensor(centers, dtype=torch.float)
                target['level_slice_classification'] = torch.as_tensor(slice_classification).long()

                        
            if self.transform is not None:
                # Need to reshape it around
                x = x.transpose(1, 2, 3, 0).reshape(self.patch_size, self.patch_size, -1)
                x = self.transform(image=x)['image'].reshape(self.patch_size, self.patch_size, self.channels, -1).transpose(3, 0, 1, 2)
        
            
            
        except Exception as e:
            for j in range(5):
                self.fails.add((study.study_id, self.series_description, j))
            target['patch_offsets'] = torch.tensor(patch_offsets, dtype=torch.float)
            target['patch_scalings'] = torch.tensor(patch_scalings, dtype=torch.float)
            target['series_ids'] = torch.tensor(series_ids, dtype=torch.long)

            if full_targets:
                centers = np.zeros((5, 2), dtype=float) - 1000
                slice_classification = np.zeros((5,self.channels), dtype=int)
                target['centers'] = torch.tensor(centers, dtype=torch.float)
                target['level_slice_classification'] = torch.as_tensor(slice_classification).long()
            pass

        if self.mode == 'test':
            study.unload()
        x = x.transpose(0, 3, 1, 2)
        return torch.tensor(x, dtype=torch.float) / 255.0, target

    def __len__(self):
        return len(self.study_ids)


class LevelCubeLeftRightAreaDataset(Dataset):
    def __init__(self, study_ids, channels: int, patch_size: int, d_side: float, series_description: str, left_condition: str, right_condition: str, generated_coordinate_file: str, mode='train', transform: callable = None, load_studies: list[OrientedStudy]|None = None):
        self.study_ids = list(study_ids)
        self.patch_size = int(patch_size)
        self.channels = channels
        self.d_side = d_side
        logger.info(f'Output will have size {self.patch_size} and {channels} channels and cover area with {d_side} mm side')
        self.mode = mode
        self.transform = transform
        
        self.label_columns = sum([[create_column(condition, level=level) for level in LEVELS] for condition in CONDITIONS if condition in [left_condition, right_condition]], [])
        assert len(self.label_columns) == 10
        self.series_description = series_description
        self.left_dataset = LevelCubeAreaDataset(study_ids=study_ids, 
                                             channels=channels, 
                                             patch_size=patch_size, 
                                             d_side=d_side,
                                             series_description=series_description, 
                                             condition=left_condition, 
                                             generated_coordinate_file=generated_coordinate_file, 
                                             mode=mode, 
                                             transform=transform, 
                                             load_studies=None)
        self.right_dataset = LevelCubeAreaDataset(study_ids=study_ids, 
                                             channels=channels, 
                                             patch_size=patch_size, 
                                             d_side=d_side,
                                             series_description=series_description, 
                                             condition=right_condition, 
                                             generated_coordinate_file=generated_coordinate_file, 
                                             mode=mode, 
                                             transform=transform, 
                                             load_studies=self.left_dataset.studies)
            
    @property
    def labels(self):
        return self.label_columns

    @property
    def fails(self):
        return self.left_dataset.fails | self.right_dataset.fails

    def __getitem__(self, idx):
        """
        Will output a tensor of size LEVELS, SIDE, CHANNELS, PATCH_SIZE, PATCH_SIZE
        """
        study = self.left_dataset.studies[idx]
        full_targets = self.mode == 'train' or self.mode == 'valid'
        target = {'study_id': study.study_id}
        if full_targets:
            label = np.int64([study.labels[col] for col in self.label_columns])
            target['labels'] = torch.tensor(label)
        left_x, left_targets = self.left_dataset[idx]
        right_x, right_targets = self.right_dataset[idx]
        if self.series_description == 'Axial T2':
            x = torch.stack([left_x, torch.flip(right_x, dims=[-1])], dim=1)
        # elif self.series_description == 'Sagittal T1':
        #     x = torch.stack([left_x, torch.flip(right_x, dims=[-3])], dim=1)
        else:
            x = torch.stack([left_x, right_x], dim=1)

        for key in left_targets.keys():
            if key not in ['study_id', 'labels']:
                left_t = left_targets[key]
                right_t = right_targets[key]
                target[key] = torch.stack([left_t, right_t], dim=1)

        return x, target
        
    def __len__(self):
        return len(self.left_dataset)


class MasterLevelCubeAreaDataset(Dataset):
    def __init__(self, study_ids, generated_coordinate_file: str, mode='train', transform: callable = None, load_studies: list[OrientedStudy]|None = None):
        self.spinal_dataset = LevelCubeAreaDataset(study_ids=study_ids, 
                                                   patch_size=224, 
                                                   channels=11, 
                                                   d_side=50.0, 
                                                   series_description='Sagittal T2/STIR', 
                                                   condition='Spinal Canal Stenosis', 
                                                   generated_coordinate_file=generated_coordinate_file, 
                                                   transform=transform, 
                                                   mode=mode,
                                                   load_studies=load_studies)
        
        self.foraminal_dataset = LevelCubeLeftRightAreaDataset(study_ids=study_ids, 
                                                   patch_size=224, 
                                                   channels=11, 
                                                   d_side=50.0, 
                                                   series_description='Sagittal T1', 
                                                   left_condition='Left Neural Foraminal Narrowing', 
                                                   right_condition='Right Neural Foraminal Narrowing', 
                                                   generated_coordinate_file=generated_coordinate_file, 
                                                   transform=transform, 
                                                   mode=mode,
                                                   load_studies=self.spinal_dataset.studies)
        
        self.subarticular_dataset = LevelCubeLeftRightAreaDataset(study_ids=study_ids, 
                                                   patch_size=224, 
                                                   channels=7, 
                                                   d_side=40.0, 
                                                   series_description='Axial T2', 
                                                   left_condition='Left Subarticular Stenosis', 
                                                   right_condition='Right Subarticular Stenosis', 
                                                   generated_coordinate_file=generated_coordinate_file, 
                                                   transform=transform, 
                                                   mode=mode,
                                                   load_studies=self.spinal_dataset.studies)
        
        self.label_columns = sum([[create_column(condition, level=level) for level in LEVELS] for condition in CONDITIONS], [])


    @property
    def mode(self):
        return self.spinal_dataset.mode
    
    @property
    def labels(self):
        return self.label_columns
    
    @property
    def fails(self):
        return self.spinal_dataset.fails | self.foraminal_dataset.fails | self.subarticular_dataset.fails

    def __getitem__(self, idx):
        """
        Will output a tensor of size LEVELS, SIDE, CHANNELS, PATCH_SIZE, PATCH_SIZE
        """
        study = self.spinal_dataset.studies[idx]
        full_targets = self.mode == 'train' or self.mode == 'valid'
        target = {'study_id': study.study_id}
        if full_targets:
            label = np.int64([study.labels[col] for col in self.label_columns])
            target['labels'] = torch.tensor(label)
        spinal_x, spinal_targets = self.spinal_dataset[idx]
        foraminal_x, foraminal_targets = self.foraminal_dataset[idx]
        subarticular_x, subarticular_targets = self.subarticular_dataset[idx]
        
        data = {}
        data['Sagittal T2/STIR Patch'] = spinal_x
        data['Sagittal T1 Patch'] = foraminal_x
        data['Axial T2 Patch'] = subarticular_x

        for key in spinal_targets.keys():
            if key not in ['study_id', 'labels']:
                target[f'Sagittal T2/STIR {key}'] = spinal_targets[key]

        for key in foraminal_targets.keys():
            if key not in ['study_id', 'labels']:
                target[f'Sagittal T1 {key}'] = foraminal_targets[key]

        for key in subarticular_targets.keys():
            if key not in ['study_id', 'labels']:
                target[f'Axial T2 {key}'] = subarticular_targets[key]

        return data, target
        
    def __len__(self):
        return len(self.spinal_dataset)
    


class LevelCubeClosestAreaDataset(Dataset):
    def __init__(self, study_ids, channels: int, patch_size: int, d_side: float, series_description: str, condition: str, generated_coordinate_file: str, mode='train', transform: callable = None, load_studies: list[OrientedStudy]|None = None):
        self.study_ids = list(study_ids)
        self.patch_size = int(patch_size)
        self.channels = int(channels)
        self.d_side = d_side
        logger.info(f'Output will have patch_size {self.patch_size} and {channels} channels, cover area with side of {d_side} mm')
        self.mode = mode
        self.transform = transform
        if self.mode == 'train' or self.mode == 'valid':
            self.labels_df, self.coordinate_df, self.series_description_df = load_train_files(relative_directory=rsnautils.relative_directory, clean=rsnautils.CLEAN)
        else:
            self.labels_df = None
            self.coordinate_df = None
            self.series_description_df = load_test_files(relative_directory=rsnautils.relative_directory)

        if load_studies is None:
            logger.info(f'Loading {len(study_ids)} Studies')
            self.studies = [OrientedStudy(study_id=study_id, labels_df=self.labels_df, series_description_df=self.series_description_df, coordinate_df=self.coordinate_df) for study_id in study_ids]
            logger.info(f'Done')
        else:
            logger.info(f'Referencing pre-loaded studies')
            self.studies = load_studies
            assert len(self.study_ids) == len(self.studies)


        self.label_columns = [create_column(condition, level=level) for level in LEVELS]
        self.series_description = series_description
        self.condition = condition
            
        self.pred_center_df = pd.read_csv(generated_coordinate_file)  # Should be a file just like any other coordinate file

        self.fails = set()

    @property
    def labels(self):
        return self.label_columns

    def __getitem__(self, idx):
        """
        Will output a tensor of size LEVELS, CHANNELS, PATCH_SIZE, PATCH_SIZE
        """
        final_size = 5, self.patch_size, self.patch_size, int(self.channels)
        x = np.zeros(final_size, dtype=np.uint8)
        full_targets = self.mode == 'train' or self.mode == 'valid'
        study = self.studies[idx]
        target = {'study_id': study.study_id}
        patch_offsets = np.zeros((5, 2), dtype=int)
        patch_scalings = np.zeros((5, 2), dtype=int)
        series_ids = np.zeros((5,), dtype=int)

        try:
            study.load()
            if full_targets:
                label = np.int64([study.labels[col] for col in self.label_columns])
                target['labels'] = torch.tensor(label)

            # Find all points from coordinate dataframe for this study
            tmp = self.pred_center_df[(self.pred_center_df.study_id == study.study_id) & (self.pred_center_df.condition == self.condition)].sort_values('level')
            # assert len(tmp) <= 5
            # For each level, find the series and extract the patch
            saves = []
            level_dict = {lev: i for i, lev in enumerate(LEVELS)}
            patch_offsets = np.zeros((5, 2), dtype=int)
            patch_scalings = np.zeros((5, 2), dtype=int)
            series_ids = np.zeros((5,), dtype=int)
            for row in tmp.itertuples():
                try:
                    series = study.get_series(row.series_id)
                    x0=int(round(row.x)) 
                    y0=int(round(row.y))
                    if self.series_description == 'Axial T2':
                        x0, y0 = y0, x0

                    inum = row.instance_number
                    if self.mode == 'train':
                        gap = int(self.patch_size // 10)
                        x0 += np.random.randint(-gap, gap+1)
                        y0 += np.random.randint(-gap, gap+1)
                        stack = series.get_stack(row.instance_number)
                        k = stack._get_instance_k(row.instance_number)
                        k = np.clip(k + np.random.randint(-1, 2), 0, stack.number_of_instances-1)
                        inum = stack.instance_numbers[k]

                    stack = series.get_stack(instance_number=inum)
                    world_x, world_y, world_z = stack.get_world_coordinates(instance_number=inum, x=x0, y=y0)
                    p0 = np.array([world_x, world_y, world_z])

                    val = []
                    distance_cutoff = self.d_side / 2
                    for v in study.series_dict.values():
                        if v.series_description == series.series_description:
                            for s in v.dicom_stacks:
                                n = s.dicom_info['affine'][:3, 2]
                                norm = n / np.sqrt(np.dot(n, n))
                                inv_affine = np.linalg.inv(s.dicom_info['affine'])
                                for i, pt in enumerate(s.dicom_info['positions']):
                                    v0 = p0 - pt
                                    distance = np.dot(v0, norm)
                                    if abs(distance) < distance_cutoff:
                                        p1 = v0 - distance*norm + pt
                                        p2 = np.ones(4)
                                        p2[:3] = p1
                                        x1, y1, z1, __ = np.dot(inv_affine, p2)
                                        # print(x,y,z)
                                        val.append([s, i, s.dicom_info['instance_number'][i], int(x1), int(y1), distance])

                    val2 = sorted(sorted(val, key=lambda xs: abs(xs[-1]))[:self.channels], key=lambda xss: x[-1])

                    lev = level_dict[row.level]
                    offset = 0
                    if len(val2) == self.channels - 2 or len(val2) == self.channels - 1:
                        offset = 1
                    elif len(val2) == self.channels - 3 or len(val2) == self.channels - 4:
                        offset = 2
                    elif len(val2) < self.channels:
                        offset = 3
                    for j, r in enumerate(val2):
                        stack, __, inum, x0, y0, __ = r
                        patch, instance_numbers, patch_offset, scaling = stack.get_thick_area(instance_number=inum, slice_thickness=1, x=x0, y=y0, patch_size=224, d_mm=40)
                        
                        x[lev, :, :, j+offset] = patch[0]
                    
                    # patch_offsets[lev, 0] = patch_offset[0]
                    # patch_offsets[lev, 1] = patch_offset[1]
                    # patch_scalings[lev, 0] = scaling[0]
                    # patch_scalings[lev, 1] = scaling[1]
                    series_ids[lev] = row.series_id if isinstance(row.series_id, int) else 0
                    saves.append((lev, row.level, series.series_id, instance_numbers, patch_offset, scaling))
                except (KeyError, ValueError):
                    self.fails.append((study.study_id, self.series_description, level_dict[row.level]))
                    pass

            target['patch_offsets'] = torch.tensor(patch_offsets, dtype=torch.float)
            target['patch_scalings'] = torch.tensor(patch_scalings, dtype=torch.float)
            target['series_ids'] = torch.tensor(series_ids, dtype=torch.long)
            if full_targets:
                study_mask = (self.coordinate_df.study_id == study.study_id) & (self.coordinate_df.condition == self.condition)
                tmp2 = self.coordinate_df.loc[study_mask]
                centers = np.zeros((5, 2), dtype=float) - 1000
                slice_classification = np.zeros((5,self.channels), dtype=int)
                for lev, level, series_id, instance_numbers, patch_offset, scaling in saves:
                    series = study.get_series(series_id)
                    used_instances = tmp2.loc[(tmp2.series_id == series_id) & (tmp2.level == level), 'instance_number'].unique()
                    
                    slice_classification[lev] = np.array([(1 if j in used_instances else 0) for j in instance_numbers], dtype=int)
                    tmp3 = tmp2.loc[(tmp2.series_id == series_id) & (tmp2.level == level), ['x', 'y']]
                    if len(tmp3) == 1:
                        x0, y0 = tmp3.iloc[0].values
                        x0 /= scaling[0]
                        y0 /= scaling[1]
                        x0 -= patch_offset[0]
                        y0 -= patch_offset[1]
                        centers[lev] = x0, y0
                target['centers'] = torch.tensor(centers, dtype=torch.float)
                target['level_slice_classification'] = torch.as_tensor(slice_classification).long()

                        
            if self.transform is not None:
                # Need to reshape it around
                x = x.transpose(1, 2, 3, 0).reshape(self.patch_size, self.patch_size, -1)
                x = self.transform(image=x)['image'].reshape(self.patch_size, self.patch_size, self.channels, -1).transpose(3, 0, 1, 2)
        
            
            
        except Exception as e:
            for j in range(5):
                self.fails.add((study.study_id, self.series_description, j))
            target['patch_offsets'] = torch.tensor(patch_offsets, dtype=torch.float)
            target['patch_scalings'] = torch.tensor(patch_scalings, dtype=torch.float)
            target['series_ids'] = torch.tensor(series_ids, dtype=torch.long)

            if full_targets:
                centers = np.zeros((5, 2), dtype=float) - 1000
                slice_classification = np.zeros((5,self.channels), dtype=int)
                target['centers'] = torch.tensor(centers, dtype=torch.float)
                target['level_slice_classification'] = torch.as_tensor(slice_classification).long()
            pass

        if self.mode == 'test':
            study.unload()
        x = x.transpose(0, 3, 1, 2)
        return torch.tensor(x, dtype=torch.float) / 255.0, target

    def __len__(self):
        return len(self.study_ids)


class LevelCubeLeftRightClosestAreaDataset(Dataset):
    def __init__(self, study_ids, channels: int, patch_size: int, d_side: float, series_description: str, left_condition: str, right_condition: str, generated_coordinate_file: str, mode='train', transform: callable = None, load_studies: list[OrientedStudy]|None = None):
        self.study_ids = list(study_ids)
        self.patch_size = int(patch_size)
        self.channels = channels
        self.d_side = d_side
        logger.info(f'Output will have size {self.patch_size} and {channels} channels and cover area with {d_side} mm side')
        self.mode = mode
        self.transform = transform
        
        self.label_columns = sum([[create_column(condition, level=level) for level in LEVELS] for condition in CONDITIONS if condition in [left_condition, right_condition]], [])
        assert len(self.label_columns) == 10
        self.series_description = series_description
        self.left_dataset = LevelCubeClosestAreaDataset(study_ids=study_ids, 
                                             channels=channels, 
                                             patch_size=patch_size, 
                                             d_side=d_side,
                                             series_description=series_description, 
                                             condition=left_condition, 
                                             generated_coordinate_file=generated_coordinate_file, 
                                             mode=mode, 
                                             transform=transform, 
                                             load_studies=None)
        self.right_dataset = LevelCubeClosestAreaDataset(study_ids=study_ids, 
                                             channels=channels, 
                                             patch_size=patch_size, 
                                             d_side=d_side,
                                             series_description=series_description, 
                                             condition=right_condition, 
                                             generated_coordinate_file=generated_coordinate_file, 
                                             mode=mode, 
                                             transform=transform, 
                                             load_studies=self.left_dataset.studies)
            
    @property
    def labels(self):
        return self.label_columns

    @property
    def fails(self):
        return self.left_dataset.fails | self.right_dataset.fails

    def __getitem__(self, idx):
        """
        Will output a tensor of size LEVELS, SIDE, CHANNELS, PATCH_SIZE, PATCH_SIZE
        """
        study = self.left_dataset.studies[idx]
        full_targets = self.mode == 'train' or self.mode == 'valid'
        target = {'study_id': study.study_id}
        if full_targets:
            label = np.int64([study.labels[col] for col in self.label_columns])
            target['labels'] = torch.tensor(label)
        left_x, left_targets = self.left_dataset[idx]
        right_x, right_targets = self.right_dataset[idx]
        if self.series_description == 'Axial T2':
            x = torch.stack([left_x, torch.flip(right_x, dims=[-1])], dim=1)
        # elif self.series_description == 'Sagittal T1':
        #     x = torch.stack([left_x, torch.flip(right_x, dims=[-3])], dim=1)
        else:
            x = torch.stack([left_x, right_x], dim=1)

        for key in left_targets.keys():
            if key not in ['study_id', 'labels']:
                left_t = left_targets[key]
                right_t = right_targets[key]
                target[key] = torch.stack([left_t, right_t], dim=1)

        return x, target
        
    def __len__(self):
        return len(self.left_dataset)