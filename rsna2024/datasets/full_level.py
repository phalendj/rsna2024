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



class FullLevelDataset(Dataset):
    """
    Idea here is that because it seems levels are somewhat correllated with their results, we would like to produce all of the information of a level in one shot
    """

    def __init__(self, study_ids, channels_sag: int, patch_size_sag: int, d_sag: float, d_slice_sag: float, channels_ax: int, patch_size_ax: int, d_ax: float, d_slice_ax:float, 
                 conditions: list[str], generated_coordinate_file: str, aug_size: float, mode='train', transform: callable = None, load_studies: list[OrientedStudy]|None = None):
        self.study_ids = list(study_ids)
        self.patch_size_sag = int(patch_size_sag)
        self.channels_sag = int(channels_sag)
        self.d_sag = d_sag
        self.d_slice_sag = d_slice_sag
        self.patch_size_ax = int(patch_size_ax)
        self.channels_ax = int(channels_ax)
        self.d_ax = d_ax
        self.d_slice_ax = d_slice_ax
        self.aug_size = aug_size
        logger.info(f'Sagittal Output will have patch_size {self.patch_size_sag} and {self.channels_sag} channels spanning {d_sag} mm')
        logger.info(f'Axial Output will have patch_size {self.patch_size_ax} and {self.channels_ax} channels spanning {d_ax} mm')
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


        self.label_columns = sum([[create_column(condition, level=level) for level in LEVELS] for condition in CONDITIONS if condition in conditions], [])
        self.series_description = 'Sagittal T2/STIR'  # Use this because it is the center
        self.condition = 'Spinal Canal Stenosis'
            
        self.center_patch = True
        self.center_slice = True

        self.pred_center_df = pd.read_csv(generated_coordinate_file)  # Should be a file just like any other coordinate file

    @property
    def labels(self):
        return self.label_columns

    def get_level(self, row, study):
        patch1 = np.zeros((self.channels_sag, self.patch_size_sag, self.patch_size_sag), dtype=np.uint8)
        instance_numbers1 = np.ones((self.channels_sag,), dtype=int)*-1
        offset1 = 0, 0
        scaling1 = np.array([1,1], dtype=float)

        patch2 = np.zeros((self.channels_sag, self.patch_size_sag, self.patch_size_sag), dtype=np.uint8)
        instance_numbers2 = np.ones((self.channels_sag,), dtype=int)*-1
        offset2 = 0, 0
        scaling2 = np.array([1,1], dtype=float)

        patch3 = np.zeros((self.channels_ax, self.patch_size_ax, self.patch_size_ax), dtype=np.uint8)
        instance_numbers3 = np.ones((self.channels_ax,), dtype=int)*-1
        offset3 = 0, 0
        scaling3 = np.array([1,1], dtype=float)

        series_ids = np.zeros((3,), dtype=int)
        try:
            series = study.get_series(row.series_id)
            series_ids[0] = row.series_id
            stack = series.get_stack(row.instance_number)

            ## TODO: Add?
            x0=int(round(row.x)) 
            y0=int(round(row.y))
            
            inum = row.instance_number
            if self.mode == 'train':
                pixel_spacing = stack.dicom_info['pixel_spacing'][0, 0]
                gap = int(self.aug_size*self.d_sag / pixel_spacing)
                x0 += np.random.randint(-gap, gap+1)
                y0 += np.random.randint(-gap, gap+1)
                k = stack._get_instance_k(row.instance_number)
                #k = np.clip(k + np.random.randint(-1, 2), 0, stack.number_of_instances-1)
                inum = stack.instance_numbers[k]

            world_x, world_y, world_z = stack.get_world_coordinates(instance_number=inum, x=x0, y=y0)
            patch1, instance_numbers1, offset1, scaling1 = stack.get_thick_volume(instance_number=inum, slice_thickness=self.channels_sag, x=x0, y=y0, patch_size=self.patch_size_sag, 
                                                                                  d_mm=self.d_sag, 
                                                                                  dx_slice_mm=self.d_slice_sag,
                                                                                  center=self.center_slice, center_patch=self.center_patch)
            sd = 'Sagittal T1'
            res = None
            bdist = 1.e10
            for series_id, series in study.series_dict.items():
                if series.series_description == sd:
                    stack, dist = series.find_closest_stack(world_x, world_y, world_z, required_in=True)
                    if dist < bdist:
                        res = stack
                        bdist = dist
                        series_ids[1] = series.series_id
            if res is not None:
                inum, x, y = res.get_instance_xy_from_world(world_x, world_y, world_z)
                patch2, instance_numbers2, offset2, scaling2 = res.get_thick_volume(instance_number=inum, slice_thickness=self.channels_sag, x=x, y=y, 
                                                                                    patch_size=self.patch_size_sag, 
                                                                                    d_mm=self.d_sag, dx_slice_mm=self.d_slice_sag,
                                                                                    center=self.center_slice, center_patch=self.center_patch)
            
            sd = 'Axial T2'
            res = None
            bdist = 1.e10
            for series_id, series in study.series_dict.items():
                if series.series_description == sd:
                    stack, dist = series.find_closest_stack(world_x, world_y, world_z, required_in=True)
                    if dist < bdist:
                        res = stack
                        bdist = dist
                        series_ids[2] = series.series_id
            if res is not None:
                inum, x, y = res.get_instance_xy_from_world(world_x, world_y, world_z)
                ## TODO: DO WE FLIP X,Y -> YES?
                x, y = y, x
                patch3, instance_numbers3, offset3, scaling3 = res.get_thick_volume(instance_number=inum, slice_thickness=self.channels_ax, x=x, y=y, patch_size=self.patch_size_ax, 
                                                                                    d_mm=self.d_ax, dx_slice_mm=self.d_slice_ax,
                                                                                    center=self.center_slice, center_patch=self.center_patch)
        except KeyError:
            # logger.error(f'Key Error on {row}')
            pass
        except Exception as e:
            logger.error(f'Error on {row}')
            logger.error(f'Error on Series {series.series_id}')
            for stack in series.dicom_stacks:
                logger.error(f'{stack}')
            logger.exception(e)
            pass

        return {'Sagittal T2/STIR Patch': patch1, 'Sagittal T2/STIR Instance Numbers': instance_numbers1, 'Sagittal T2/STIR Offsets': np.array(offset1), 'Sagittal T2/STIR Scalings': scaling1,
                'Sagittal T1 Patch': patch2, 'Sagittal T1 Instance Numbers': instance_numbers2, 'Sagittal T1 Offsets': np.array(offset2), 'Sagittal T1 Scalings': scaling2,
                'Axial T2 Patch': patch3, 'Axial T2 Instance Numbers': instance_numbers3, 'Axial T2 Offsets': np.array(offset3), 'Axial T2 Scalings': scaling3, 'series_ids': series_ids}

    def __getitem__(self, idx):
        """
        Will output a tensor of size LEVELS, CHANNELS, PATCH_SIZE, PATCH_SIZE
        """
        study = self.studies[idx]
        study.load()
        full_targets = self.mode == 'train' or self.mode == 'valid'
        target = {'study_id': torch.tensor([study.study_id])}
        data = {'study_id': np.array([study.study_id])}
        if full_targets:
            label = np.int64([study.labels[col] for col in self.label_columns])
            target['labels'] = torch.tensor(label)

        # Find all points from coordinate dataframe for this study
        tmp = self.pred_center_df[(self.pred_center_df.study_id == study.study_id) & (self.pred_center_df.condition == self.condition)].sort_values('level')
        # assert len(tmp) <= 5
        # For each level, find the series and extract the patch
        saves = []
        level_dict = {lev: i for i, lev in enumerate(LEVELS)}
        for i, lev in enumerate(LEVELS):
            t = tmp[tmp.level == lev]
            if len(t) > 0:
                row = t.iloc[0]
            else:
                if len(tmp) > 0:
                    # logger.error(f'Nothing for level {lev} and {study.study_id}')
                    row = tmp.iloc[0]
                else:
                    # logger.error(f'Nothing for {study.study_id}')
                    row = self.pred_center_df.iloc[0]
            saves.append(self.get_level(row, study))
            
        for key in saves[0].keys():
            data[key] = np.stack([d[key] for d in saves]) 

        if full_targets:
            study_mask = (self.coordinate_df.study_id == study.study_id)
            tmp2 = self.coordinate_df.loc[study_mask]

            centers = {c: np.zeros((5, 2), dtype=float) - 1 for c in CONDITIONS}
            instance_numbers = {c: np.zeros((5, self.channels_ax if 'Subarticular' in c else self.channels_sag), dtype=int) for c in CONDITIONS}

            for condition in CONDITIONS:
                j = 0
                nm = 'Sagittal T2/STIR'
                if 'Foraminal' in condition:
                    j = 1
                    nm = 'Sagittal T1'
                elif 'Subarticular' in condition:
                    j = 2
                    nm = 'Axial T2'
                for i, level in enumerate(LEVELS):
                    
                    series_id = data['series_ids'][i, j]
                    mask = (tmp2.series_id == series_id) & (tmp2.condition == condition) & (tmp2.level == level)
                    t = tmp2[mask]
                    if len(t) > 0:
                        row2 = t.iloc[0]
                        used_instances = {row2.instance_number}
                        tmp_instance_numbers = data[f'{nm} Instance Numbers'][i]
                        instance_numbers[condition][i] = np.array([(1 if j in used_instances else 0) for j in tmp_instance_numbers], dtype=int)

                        centers[condition][i] = np.array([row2.x, row2.y])

            target.update({f'{k} Centers': torch.tensor(v, dtype=torch.float) for k, v in centers.items()})
            target.update({f'{k} Slice Classification': torch.tensor(v, dtype=torch.float) for k, v in instance_numbers.items()})
                    
        if self.transform is not None:
            # Need to reshape it around
            key = 'Sagittal T2/STIR Patch'
            channels = self.channels_sag
            patch_size = self.patch_size_sag
            x = data[key]
            x = x.transpose(2, 3, 0, 1).reshape(patch_size, patch_size, 5*channels)
            x = self.transform(image=x)['image'].reshape(patch_size, patch_size, 5, channels).transpose(2, 3, 0, 1)
            data[key] = x

            key = 'Sagittal T1 Patch'
            channels = self.channels_sag
            patch_size = self.patch_size_sag
            x = data[key]
            x = x.transpose(2, 3, 0, 1).reshape(patch_size, patch_size, 5*channels)
            x = self.transform(image=x)['image'].reshape(patch_size, patch_size, 5, channels).transpose(2, 3, 0, 1)
            data[key] = x

            key = 'Axial T2 Patch'
            channels = self.channels_ax
            patch_size = self.patch_size_ax
            x = data[key]
            x = x.transpose(2, 3, 0, 1).reshape(patch_size, patch_size, 5*channels)
            x = self.transform(image=x)['image'].reshape(patch_size, patch_size, 5, channels).transpose(2, 3, 0, 1)
            data[key] = x

        # convert data to tensors
        for key in data.keys():
            x = data[key]
            if 'Patch' in key:
                data[key] = torch.tensor(x, dtype=torch.float) / 255.0
            else:
                data[key] = torch.tensor(x).long()

        if self.mode == 'test':
            study.unload()

        return data, target

    def __len__(self):
        return len(self.study_ids)
    