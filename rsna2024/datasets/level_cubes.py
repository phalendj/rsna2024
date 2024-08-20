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
    from utils import relative_directory
    import utils as rsnautils
    from datasets import load_train_files, load_test_files, LEVELS, CONDITIONS, create_column
    from datasets.dicom_load import OrientedStudy
except ImportError:
    from ..utils import relative_directory
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
            self.labels_df, self.coordinate_df, self.series_description_df = load_train_files(relative_directory=relative_directory, clean=rsnautils.CLEAN)
        else:
            self.labels_df = None
            self.coordinate_df = None
            self.series_description_df = load_test_files(relative_directory=relative_directory)

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
            except KeyError:
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
            self.labels_df, self.coordinate_df, self.series_description_df = load_train_files(relative_directory=relative_directory, clean=rsnautils.CLEAN)
        else:
            self.labels_df = None
            self.coordinate_df = None
            self.series_description_df = load_test_files(relative_directory=relative_directory)

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


### TODO: If needed, use the above and add a resize to the cubes extracted
# class LevelCubeCropZoomDataset(Dataset):
#     def __init__(self, study_ids, channels: int, image_size: tuple[int, int], slice_size: int, series_description, conditions, mode='train', transform=None, load_studies=None):
#         self.study_ids = list(study_ids)
#         self.slice_size = int(slice_size)
#         self.image_size = int(image_size[0]), int(image_size[1])
#         self.channels = channels
#         logger.info(f'Output will have size {2*self.slice_size} resized to {image_size} and {channels} channels')
#         self.mode = mode
#         self.transform = transform
#         if self.mode == 'train' or self.mode == 'valid':
#             self.labels_df, self.coordinate_df, self.series_description_df = load_train_files(relative_directory=relative_directory, clean=CLEAN)
#         else:
#             self.labels_df = None
#             self.coordinate_df = None
#             self.series_description_df = load_test_files(relative_directory=relative_directory)

#         if load_studies is None:
#             logger.info(f'Loading {len(study_ids)} Studies')
#             self.studies = [Study(study_id=study_id, labels_df=self.labels_df, series_description_df=self.series_description_df, coordinate_df=self.coordinate_df) for study_id in study_ids]
#             logger.info(f'Done')
#         else:
#             logger.info(f'Referencing pre-loaded studies')
#             self.studies = load_studies
#             assert len(self.study_ids) == len(self.studies)


#         self.label_columns = sum([[create_column(condition, level=level) for level in LEVELS] for condition in CONDITIONS if condition in conditions], [])
#         self.series_description = series_description

#         if self.mode == 'train' or self.mode == 'valid':
#             series2cond = {'Sagittal T2/STIR': 'spinal',  'Sagittal T1': 'foraminal', 'Axial T2': 'subarticular'}
#             self.available_diagnosis = [c for c in self.label_columns if series2cond[self.series_description] in c]
            

#     @property
#     def labels(self):
#         return self.label_columns

#     def __getitem__(self, idx):
#         """
#         Will output a tensor of size LEVELS, CHANNELS, 2*SLICE_SIZE, SLICE_SIZE
#         """
#         final_size = 5, self.image_size[0], self.image_size[1], int(self.channels)
#         x = np.zeros(final_size, dtype=np.uint8)
#         study = self.studies[idx]
#         if self.mode == 'train' or self.mode == 'valid':
#             label = np.int64([study.labels[col] for col in self.label_columns])
#         else:
#             label = np.int64([-100 for col in self.label_columns])

#         available = [s[2] for s in study.series if s[1] == self.series_description]
#         if len(available) > 0:
#             if self.mode == 'train':
#                 series = np.random.choice(available)
#             else:
#                 series = available[0]
            
#             data = series.data
#             data = data.transpose(1, 2, 0)
#             H, W, D = data.shape

#             for k in range(5):
#                 xr = []
#                 yr = []
#                 zr = []
#                 k2 = k
#                 while k2 < len(self.available_diagnosis):

#                     nm = self.available_diagnosis[k2]
#                     if nm in series.diagnosis_coordinates:
#                         y0, x0, z0 = series.diagnosis_coordinates[nm]
#                         xr.append(x0)
#                         yr.append(y0)
#                         zr.append(z0)
#                     k2 += 5

#                 if len(xr) > 0:
#                     x0 = np.mean(xr)
#                     y0 = np.mean(yr)
#                     z0 = np.mean(zr)
#                     x0 = min(max(int(x0) - self.slice_size, 0), H - 2*self.slice_size)
#                     y0 = min(max(int(y0) - self.slice_size, 0), W - 2*self.slice_size)
#                     x1 = min(H, x0 + 2*self.slice_size)
#                     y1 = min(W, y0 + 2*self.slice_size)

#                     i0 = max(int(z0 - self.channels/ 2), 0)
#                     i1 = min(i0 + self.channels, D)
#                     # Axial images can have smaller pixel sizes, and when doing levels we really want the entire image
#                     if self.series_description == 'Axial T2' and (2*self.slice_size > H or 2*self.slice_size > W):
#                         data2 = data[..., i0:i1].copy()
#                         if H > W:
#                             diff = H-W
#                             if self.mode == 'train':
#                                 offset = np.random.randint(diff)
#                             else:
#                                 offset = int(diff//2)
#                             data2 = data2[offset:offset+W]
#                             H = W
#                         elif W > H:
#                             diff = W-H
#                             if self.mode == 'train':
#                                 offset = np.random.randint(diff)
#                             else:
#                                 offset = int(diff//2)

#                             data2 = data2[:, offset:offset+H]
#                             W = H
#                         data2 = cv2.resize(data2, self.image_size, interpolation=cv2.INTER_LANCZOS4)
#                         x[k, ..., :(i1-i0)] = data2
                        
#                     else:
#                         x[k, :, :, :(i1-i0)] = cv2.resize(data[x0:x1, y0:y1, i0:i1], self.image_size, interpolation=cv2.INTER_LANCZOS4)
                    
#             if self.transform is not None:
#                 # Need to reshape it around
#                 x = x.transpose(1, 2, 3, 0).reshape(*self.image_size, -1)
#                 x = self.transform(image=x)['image'].reshape(*self.image_size, self.channels, -1).transpose(3, 0, 1, 2)
        
#         x = x.transpose(0, 3, 1, 2)

#         target = {}
#         target['labels'] = torch.tensor(label)
#         target['study_id'] = torch.tensor([study.study_id])

#         return torch.tensor(x, dtype=torch.float) / 255.0, target

#     def __len__(self):
#         return len(self.study_ids)
