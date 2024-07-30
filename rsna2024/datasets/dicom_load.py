"""
At the moment, this is a simple setup for how we analyze the different images.  

It works reasonably well for sagittal cuts, since they all have an identical orientation.  

For axial, there are different sets of images taken with slightly different angles to get better cross sections of the spine.  These are combined in a simplistic way here, 
but there is a better method we can use that takes into account the orientation.  This can be cross referenced in world space with the sagittal views to get a better axial model.  See below.


TODO:
Following:
https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/522956

We can use the different angles to better put together the axial images.  
We can also use the world coordinates, matched with some segmentation information from the sagittal cuts to figure out exactly which of the levels an image corresponds to.

"""


import os
import glob
import pickle

import cv2
import numpy as np
import pandas as pd

import pydicom

from utils import image_directory


def get_study_directory(study_id):
    return f'{image_directory}/{study_id}/'

def get_series_directory(study_id, series_id):
    return f'{image_directory}/{study_id}/{series_id}/'


def convert_to_8bit(x):
    """
    Courtesy of https://www.kaggle.com/code/vaillant/cross-reference-images-in-different-mri-planes?scriptVersionId=181874384
    """
    lower, upper = np.percentile(x, (1, 99))
    x = np.clip(x, lower, upper)
    x = x - np.min(x)
    x = x / np.max(x) 
    return (x * 255).astype("uint8")


def load_dicom_stack(dicom_folder, plane, reverse_sort=False):
    """
    Courtesy of https://www.kaggle.com/code/vaillant/cross-reference-images-in-different-mri-planes?scriptVersionId=181874384
    """
    dicom_files = glob.glob(os.path.join(dicom_folder, "*.dcm"))
    dicoms = [pydicom.dcmread(f) for f in dicom_files]
    plane = {"sagittal": 0, "coronal": 1, "axial": 2}[plane.lower()]
    positions = np.asarray([float(d.ImagePositionPatient[plane]) for d in dicoms])
    # if reverse_sort=False, then increasing array index will be from RIGHT->LEFT and CAUDAL->CRANIAL
    # thus we do reverse_sort=True for axial so increasing array index is craniocaudal
    idx = np.argsort(-positions if reverse_sort else positions)
    ipp = np.asarray([d.ImagePositionPatient for d in dicoms]).astype("float")[idx]
    iop = np.asarray([d.ImageOrientationPatient for d in dicoms]).astype("float")[idx].reshape((len(idx), 2, 3))
    instance_index = np.asarray([int(d.InstanceNumber) for d in dicoms]).astype(int)[idx]
    ps = np.asarray([d.PixelSpacing for d in dicoms]).astype("float")[idx]
    shps = np.asarray([d.pixel_array.shape for d in dicoms]).astype("float")[idx]
    all_shps = np.unique(shps, axis=0)
    if len(all_shps) > 1:
        wv = ps*shps
        assert np.all(wv.max(axis=0) - wv.min(axis=0) < ps.min(axis=0))
        image_size = int(shps[0, 0]), int(shps[0, 1])
        for i in range(len(ps)):
            ps[i] = ps[0]
        array = np.stack([cv2.resize(d.pixel_array, image_size, interpolation=cv2.INTER_LANCZOS4).astype("float32") for d in dicoms])
    else:
        array = np.stack([d.pixel_array.astype("float32") for d in dicoms])
    array = array[idx]
    
    return {"array": convert_to_8bit(array), "positions": ipp, "pixel_spacing": ps, "orientations": iop, 'instance_number': instance_index}



class Series(object):
    def __init__(self, path_to_dicom, labels_df, series_description_df, coordinate_df):
        self.path_to_dicom = path_to_dicom
        self.files = sorted([f for f in os.listdir(path_to_dicom) if '.dcm' in f], key=lambda s: int(s.replace('.dcm', '')))
        self.study_id, self.series_id = [x for x in path_to_dicom.split('/') if len(x) > 0][-2:]
        self.study_id = int(self.study_id)
        self.series_id = int(self.series_id)
        if labels_df is not None:
            self.labels = labels_df[labels_df.study_id == self.study_id].iloc[0].to_dict()
            for c in ['study_id', 'stratum', 'fold']:
                if c in self.labels:
                    del self.labels[c]

        self.series_description = series_description_df.loc[(series_description_df.study_id == self.study_id) & (series_description_df.series_id == self.series_id), 'series_description'].iloc[0]
        
        if coordinate_df is not None:
            self.load()
            tmp = coordinate_df[(coordinate_df.study_id == self.study_id) & (coordinate_df.series_id == self.series_id)].copy()
            index_instance_dict = {ii: i for i, ii in enumerate(self.dicom_info['instance_number'])}
            
            tmp['z'] = tmp.instance_number.map(lambda s: index_instance_dict[s])
            
            self.diagnosis_coordinates = {}
            for __, row in tmp.iterrows():
                self.diagnosis_coordinates[row.condition.lower().replace(' ', '_') + row.level.lower().replace('/','_')] = (row.x, row.y, row.z)
        
    @property
    def data(self):
        self.load()
        return self.dicom_info['array']
            
    def unload(self):
        del self.dicom_info
        
    def load(self):
        if hasattr(self, 'dicom_info'):
            return
        
        if os.path.exists(self.path_to_dicom + '/saved.pkl'):
            with open(self.path_to_dicom + '/saved.pkl', 'rb') as f:
                self.dicom_info = pickle.load(f)
        else:
            if self.series_description == "Sagittal T2/STIR":
                self.dicom_info = load_dicom_stack(self.path_to_dicom, plane="sagittal")
            elif self.series_description == "Sagittal T1":
                self.dicom_info = load_dicom_stack(self.path_to_dicom, plane="sagittal")
            elif self.series_description == "Axial T2":
                self.dicom_info = load_dicom_stack(self.path_to_dicom, plane="axial", reverse_sort=True)
                
            with open(self.path_to_dicom + '/saved.pkl', 'wb') as f:
                pickle.dump(self.dicom_info, f)
            
    def __repr__(self):
        return f'Series(study_id={self.study_id}, series_id={self.series_id}, series_description={self.series_description})'
    

class Study(object):
    def __init__(self, study_id, labels_df, series_description_df, coordinate_df):
        self.study_id = study_id
        study = series_description_df[series_description_df.study_id == study_id]
        
        self.series = []
        for row in study.itertuples():
            s = Series(get_series_directory(row.study_id, row.series_id), labels_df=labels_df, series_description_df=series_description_df, coordinate_df=coordinate_df)
            self.series.append((row.series_id, row.series_description, s))
        
        if labels_df is not None:
            self.labels = labels_df[labels_df.study_id == self.study_id].iloc[0].to_dict()
            for c in ['study_id', 'stratum', 'fold']:
                if c in self.labels:
                    del self.labels[c]
                    
                    
    def __repr__(self):
        return f'Study(study_id={self.study_id}, n_series={len(self.series)})'