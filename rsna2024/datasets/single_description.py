import logging
import numpy as np

import cv2
import torch
from torch.utils.data import Dataset

from utils import relative_directory
from datasets import load_train_files, load_test_files, LEVELS, CONDITIONS, create_column
from datasets.dicom_load import Study

logger = logging.getLogger(__name__)



class SpinalCanalStenosisCenterDataset(Dataset):
    def __init__(self, study_ids, image_size, channels, mode='train', transform=None):
        self.study_ids = list(study_ids)
        self.image_size = int(image_size[0]), int(image_size[1])
        self.channels = channels
        logger.info(f'Output will have size {self.image_size} with {self.channels}')
        self.mode = mode
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

        condition = [c for c in CONDITIONS if 'Spinal' in c][0]
        self.label_columns = [create_column(condition, level=level) for level in LEVELS]
        self.series_description = 'Sagittal T2/STIR'

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
                data = np.random.choice(available).data
            else:
                data = available[0].data

            # Trim to appropriate size
            # logger.info(f'Data Shape : {data.shape}')
            data = data.transpose(1, 2, 0)
            # logger.info(f'Data Shape : {data.shape}')
            H, W, D = data.shape
            if H > W:
                diff = H-W
                if self.mode == 'train':
                    offset = np.random.randint(diff)
                else:
                    offset = int(diff//2)
                data = data[offset:offset+W]
                data = cv2.resize(data, self.image_size, interpolation=cv2.INTER_LANCZOS4)
            elif W > H:
                diff = W-H
                if self.mode == 'train':
                    offset = np.random.randint(diff)
                else:
                    offset = int(diff//2)

                data = data[:, offset:offset+H]
            # logger.info(f'Data Shape : {data.shape}')
            data = cv2.resize(data, self.image_size, interpolation=cv2.INTER_LANCZOS4)

            # Select the middle portion number of channels
            i0 = int((D - self.channels) // 2)
            for i in range(self.channels):
                j = i0 + i
                x[..., i] = data[..., j]

            if self.transform is not None:
                x = self.transform(image=x)['image']

        x = x.transpose(2, 0, 1)
                
        target = {}
        target['labels'] = torch.tensor(label)

        return torch.tensor(x, dtype=torch.float) / 255.0, target

    def __len__(self):
        return len(self.study_ids)