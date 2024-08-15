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
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd

import pydicom

try:
    from utils import image_directory
    from datasets import create_column
except ImportError:
    from ..utils import image_directory
    from ..datasets import create_column


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


def load_dicom_stack(dicoms: list, plane: str, reverse_sort: bool = False):
    """
    Courtesy of https://www.kaggle.com/code/vaillant/cross-reference-images-in-different-mri-planes?scriptVersionId=181874384
    """
    
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
    #TODO: Adjust centers for these issues
    if len(all_shps) > 1:
        wv = ps*shps
        assert np.all(wv.max(axis=0) - wv.min(axis=0) < ps.min(axis=0))
        image_size = int(shps[0, 0]), int(shps[0, 1])
        # Adjusting pixel spacing for the reshape
        for i in range(len(ps)):
            ps[i] = ps[0]
        array = np.stack([cv2.resize(d.pixel_array, image_size, interpolation=cv2.INTER_LANCZOS4).astype("float32") for d in dicoms])
    else:
        array = np.stack([d.pixel_array.astype("float32") for d in dicoms])
    array = array[idx]
    
    return {"array": convert_to_8bit(array), "positions": ipp, "pixel_spacing": ps, "orientations": iop, 'instance_number': instance_index}

def read_dicoms(dicom_folder):
    dicom_files = glob.glob(os.path.join(dicom_folder, "*.dcm"))
    dicoms = [pydicom.dcmread(f) for f in dicom_files]
    return dicoms


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
                self.diagnosis_coordinates[create_column(row.condition, row.level)] = (row.x, row.y, row.z)
        
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
                dicoms = read_dicoms(self.path_to_dicom)
                self.dicom_info = load_dicom_stack(dicoms, plane="sagittal")
            elif self.series_description == "Sagittal T1":
                dicoms = read_dicoms(self.path_to_dicom)
                self.dicom_info = load_dicom_stack(dicoms, plane="sagittal")
            elif self.series_description == "Axial T2":
                dicoms = read_dicoms(self.path_to_dicom)
                self.dicom_info = load_dicom_stack(dicoms, plane="axial", reverse_sort=True)
                
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
    


def construct_affine(pixel_spacing, ipp, iop, slice_spacing):
    affine = np.zeros((4,4), dtype=float)
    affine[3,3] = 1
    affine[0,2] = slice_spacing[0]
    affine[1,2] = slice_spacing[1]
    affine[2,2] = slice_spacing[2]
    affine[0, 3] = ipp[0]
    affine[1, 3] = ipp[1]
    affine[2, 3] = ipp[2]
    affine[0, 0] = iop[0][0]*pixel_spacing[0]
    affine[1, 0] = iop[0][1]*pixel_spacing[0]
    affine[2, 0] = iop[0][2]*pixel_spacing[0]
    affine[0, 1] = iop[1][0]*pixel_spacing[1]
    affine[1, 1] = iop[1][1]*pixel_spacing[1]
    affine[2, 1] = iop[1][2]*pixel_spacing[1]
    return affine



def load_dicom_stack_2(dicoms, plane: str, reverse_sort=False) -> dict:
    """
    Courtesy of https://www.kaggle.com/code/vaillant/cross-reference-images-in-different-mri-planes?scriptVersionId=181874384
    """
    plane = {"sagittal": 0, "coronal": 1, "axial": 2}[plane.lower()]
    positions = np.asarray([float(d.ImagePositionPatient[plane]) for d in dicoms])
    # if reverse_sort=False, then increasing array index will be from RIGHT->LEFT and CAUDAL->CRANIAL
    # thus we do reverse_sort=True for axial so increasing array index is craniocaudal
    idx = np.argsort(-positions if reverse_sort else positions)
    ipp = np.asarray([d.ImagePositionPatient for d in dicoms]).astype("float")[idx]
    iop = np.asarray([d.ImageOrientationPatient for d in dicoms]).astype("float")[idx].reshape((len(idx), 2, 3))
    instance_index = np.asarray([int(d.InstanceNumber) for d in dicoms]).astype(int)[idx]
    ps = np.asarray([d.PixelSpacing for d in dicoms]).astype("float")[idx]
    array = np.stack([d.pixel_array.astype("float32") for d in dicoms])
    array = array[idx]
    
    slice_spacing = np.asarray([d.SpacingBetweenSlices for d in dicoms]).astype("float")[idx]
    if len(ipp) > 1:
#         x = np.diff(ipp, axis=0)
#         if len(x) == 0:
#             print(ipp)
#             print(x)
        ss = np.diff(ipp, axis=0)[0]
    else:
        # It does not matter, we just need to make sure the affine matrix is invertable.  We can do this by making a unit vector that is orthogonal to the iop
        ss = np.cross(iop[0][0], iop[0][1])
    affine = construct_affine(pixel_spacing=ps[0], ipp=ipp[0], iop=iop[0], slice_spacing=ss)
    
    return {"array": convert_to_8bit(array), "positions": ipp, "pixel_spacing": ps, "orientations": iop, 'instance_number': instance_index, 'slice_spacing': slice_spacing, 'affine': affine}

class OrientedStack:
    def __init__(self, dicoms: list, plane: str, reverse_sort: bool = True):
        self.dicom_info = load_dicom_stack_2(dicoms=dicoms, plane=plane, reverse_sort=reverse_sort)
        self.plane = plane
        if plane == 'sagittal':
            self.dicom_info['array'] = self.dicom_info['array'].transpose(0, 2, 1)

    @property
    def data(self):
        return self.dicom_info['array']
    
    def contains_world_point(self, world_x, world_y, world_z):
        """
        returns true if closest pixel is in space
        """
        k, i, j = self.get_pixel_from_world(world_x, world_y, world_z)
        Z, X, Y = self.data.shape
        return 0 <= k < Z and 0 <= i < X and 0 <= j < Y
    
    def distance_to_center(self, world_x, world_y, world_z):
        affine = self.dicom_info['affine']
        v1 = np.array([0,0,0, 1])
        Z, X, Y = self.data.shape
        v2 = np.array([X-1,Y-1,Z-1, 1])
        affine = self.dicom_info['affine']
        world_x1, world_y1, world_z1, _1_ = (np.dot(affine, v1) + np.dot(affine, v2))/2
        return np.sqrt((world_x - world_x1)**2 + (world_y - world_y1)**2 + (world_z - world_z1)**2)
        
    @property
    def instance_numbers(self):
        return self.dicom_info['instance_number']

    @property
    def number_of_instances(self):
        return len(self.dicom_info['instance_number'])
    
    def has_instance(self, instance_number: int) -> bool:
        return instance_number in self.dicom_info['instance_number']
    
    def _get_instance_k(self, instance_number):
        if self.has_instance(instance_number=instance_number):
            k = np.argwhere(self.dicom_info['instance_number']==instance_number)[0, 0]
            return k
        else:
            raise ValueError(f"Instance {instance_number} not found in {self.dicom_info['instance_number']}")
            
    def get_instance(self, instance_number: int) -> np.array:
        k = self._get_instance_k(instance_number)
        return self.data[k]
            
    def get_world_coordinates(self, instance_number: int, x: float, y: float) -> tuple[float, float, float]:
        """
        Given pixels defined such as in train_coordinates, return the world coordinates, x, y, z
        """
        z = self._get_instance_k(instance_number)
        v = np.array([x,y,z, 1])
        affine = self.dicom_info['affine']
        world_x, world_y, world_z, _1_ = np.dot(affine, v)
        return world_x, world_y, world_z  
    
    def get_pixel_coordinates(self, instance_number: int, x: float, y: float) -> tuple[float, float, float]:
        """
        Given instance and x, y, return the pixel of the data array
        """
        k = self._get_instance_k(instance_number)
        i = int(np.round(x))
        j = int(np.round(x))
        return k, i, j
        
    def get_pixel_from_world(self, world_x, world_y, world_z) -> tuple[int, int, int]:
        """Given the world coordinates, return the pixel of the data array"""
        v = np.array([world_x, world_y, world_z, 1])
        affine = self.dicom_info['affine']
        i, j, k, _1_ = np.round(np.dot(np.linalg.inv(affine), v))
        return int(k), int(i), int(j)
    
    def get_thick_slice(self, instance_number: int, slice_thickness: int, boundary_instance: int|None = None, center: bool = False) -> tuple[np.array, np.array]:
        Z, X, Y = self.data.shape
        
        k = self._get_instance_k(instance_number)
        k0 = max(k - slice_thickness//2, 0)
        if center:
            k1 = min(Z, k + slice_thickness//2 + 1)
        else:
            k1 = min(Z, k0 + slice_thickness)
            k0 = max(0, k1 - slice_thickness)
        
        i0 = 0
        if boundary_instance is not None:
            kb = self._get_instance_k(boundary_instance)
            assert kb != k, f"When using boundary instances, we need to know which side {boundary_instance}"
            if kb < k:
                if k0 <= kb:
                    if center:
                        i0 += kb-k0 + 1
                    else:
                        k1 = min(kb+1+slice_thickness, Z)
                k0 = kb + 1
                
            else:
                k1 = min(k1, kb)
                if not center:
                    k0 = max(0, k1 - slice_thickness)
                
        x = np.zeros((slice_thickness, X, Y), dtype=self.data.dtype)
        x[i0:(i0+k1-k0)] = self.data[k0:k1]

        instances = list(self.dicom_info['instance_number'][k0:k1])
        if i0 > 0:
            instances = [-1]*i0 + instances
        if len(instances) < slice_thickness:
            instances = instances + [-1]*(slice_thickness - len(instances))

        return x, np.array(instances, dtype=int)
    
    def get_thick_patch(self, instance_number: int, slice_thickness: int, x: int, y: int, patch_size: int, boundary_instance: int|None = None, center: bool = False, center_patch: bool = False) -> tuple[np.array, np.array]:
        thick_slice, instance_numbers = self.get_thick_slice(instance_number=instance_number, slice_thickness=slice_thickness, boundary_instance=boundary_instance, center=center)
        _1_, X, Y = thick_slice.shape
        assert patch_size < X
        assert patch_size < Y
        i0 = 0
        j0 = 0
        if center_patch:
            x0 = x - patch_size //2
            x1 = x0 + patch_size
            if x0 < 0:
                i0 -= x0
                x0 = 0
            elif x1 >= X:
                x1 = X
                
            y0 = y - patch_size //2
            y1 = y0 + patch_size
            if y0 < 0:
                j0 -= y0
                y0 = 0
            elif y1 >= Y:
                y1 = Y
            
        else:
            x0 = x - patch_size //2
            x1 = x0 + patch_size
            if x0 < 0:
                x0 = 0
                x1 = x0 + patch_size
            elif x0 + patch_size >= X:
                x1 = X
                x0 = x1 - patch_size
                
            y0 = y - patch_size //2
            y1 = y0 + patch_size
            if y0 < 0:
                y0 = 0
                y1 = y0 + patch_size
            elif y0 + patch_size >= Y:
                y1 = Y
                y0 = y1 - patch_size
            
            i0 = 0
            j0 = 0
        
        patch = np.zeros((slice_thickness, patch_size, patch_size), dtype=thick_slice.dtype)
        patch[:, i0:(i0+x1-x0), j0:(j0+y1-y0)] = thick_slice[:, x0:x1, y0:y1]
        return patch, instance_numbers
            
    def plot_instance(self, instance_number: int, pts: list[tuple[float, float, str]]|None):
        fig, axes = plt.subplots(1,1, figsize=(15, 15))
        ax = axes
        k = self._get_instance_k(instance_number)
        ax.imshow(self.data[k], cmap='gray')
        if pts is not None:
            for x, y, marker in pts:
                ax.plot(x, y, marker)
        

def get_dicom_groupings(dicom_folder, plane, reverse_sort):
    dicom_files = glob.glob(os.path.join(dicom_folder, "*.dcm"))
    dicoms = [pydicom.dcmread(f) for f in dicom_files]
    groupings = defaultdict(list)
    for d in dicoms:
        groupings[str(np.round(d.ImageOrientationPatient, 5))].append(d)  # saved as floats, 6 decimal places max
    stack = [OrientedStack(dicoms, plane, reverse_sort) for __, dicoms in groupings.items()]
    return stack


class OrientedSeries(object):
    def __init__(self, path_to_dicom, series_description):
        self.path_to_dicom = path_to_dicom
        self.files = sorted([f for f in os.listdir(path_to_dicom) if '.dcm' in f], key=lambda s: int(s.replace('.dcm', '')))
        self.study_id, self.series_id = [x for x in path_to_dicom.split('/') if len(x) > 0][-2:]
        self.study_id = int(self.study_id)
        self.series_id = int(self.series_id)
        
        self.series_description = series_description

    def get_plane(self):
        return self.series_description.split()[0].lower()

    # @property
    # def data(self):
    #     self.load()
    #     return self.dicom_info['array']
            
    def unload(self):
        del self.dicom_stacks
        
    def load(self):
        if hasattr(self, 'dicom_info'):
            return
        
        if os.path.exists(self.path_to_dicom + '/saved_oriented.pkl'):
            with open(self.path_to_dicom + '/saved_oriented.pkl', 'rb') as f:
                self.dicom_stacks = pickle.load(f)
        else:
            plane = self.get_plane()
            self.dicom_stacks = get_dicom_groupings(self.path_to_dicom, plane=plane, reverse_sort=(plane == 'axial'))

#             with open(self.path_to_dicom + '/saved_oriented.pkl', 'wb') as f:
#                 pickle.dump(self.dicom_stacks, f)
            
    def get_stack(self, instance_number: int) -> np.array:
        for stack in self.dicom_stacks:
            if stack.has_instance(instance_number):
                return stack
        raise ValueError(f'Instance number {instance_number} not found in series {self.series_id}')

    def get_thick_slice(self, instance_number: int, slice_thickness: int, boundary_instance: int|None = None, center: bool = False) -> np.array:
        """
        Will return a thick slice containing instance number with the thickness.

        If `center` is false, it will take the most data possible.  If true, it will only pad either side so that the specified instance is centered
        If `boundary_instance` is specified, it will include only instances containing the specified instance up to the boundary instance, and pad after that.  Better if we are going for left or right foraminal.
        """
        return self.get_stack(instance_number).get_thick_slice(instance_number=instance_number, slice_thickness=slice_thickness, boundary_instance=boundary_instance, center=center)
    
    def get_thick_patch(self, instance_number, slice_thickness, x, y, patch_size, boundary_instance: int|None = None, center: bool = False, center_patch: bool = False) -> np.array:
        return self.get_stack(instance_number).get_thick_patch(instance_number=instance_number, slice_thickness=slice_thickness, x=x, y=y, patch_size=patch_size, boundary_instance=boundary_instance, center=center, center_patch=center_patch)
    
    def find_closest_stack(self, world_x: float, world_y: float, world_z: float, required_in) -> tuple[OrientedStack|None, float]:
        """
        Given world coordinates, will find the instance number that has the coordinates closest to the center of the image.  Also will return the distance in case we are comparing different series
        """
        best_distance = 1.e10
        best_stack = None
        for stack in self.dicom_stacks:
            if not required_in or stack.in_space(world_x, world_y, world_z):
                d = stack.distance_to_center(world_x, world_y, world_z)
                if d < best_distance:
                    best_stack = stack
        
        return best_stack, best_distance

    def get_largest_stack(self):
        """Will find the largest stack"""
        largest = None
        most_slices = 0
        for stack in self.dicom_stacks:
            if stack.number_of_instances > most_slices:
                most_slices = stack.number_of_instances
                largest = stack
        return largest

    def __repr__(self):
        return f'OrientedSeries(study_id={self.study_id}, series_id={self.series_id}, series_description={self.series_description})'
    

class OrientedStudy(object):
    def __init__(self, study_id, series_description_df, labels_df: pd.DataFrame|None = None, coordinate_df: pd.DataFrame|None = None):
        self.study_id = study_id
        study = series_description_df[series_description_df.study_id == study_id]
        
        self.series = []
        for row in study.itertuples():
            s = OrientedSeries(get_series_directory(row.study_id, row.series_id), series_description_df=row.series_description)
            self.series.append((row.series_id, row.series_description, s))
        
        if labels_df is not None:
            self.labels = labels_df[labels_df.study_id == self.study_id].iloc[0].to_dict()
            for c in ['study_id', 'stratum', 'fold']:
                if c in self.labels:
                    del self.labels[c]
                    
        self.set_coordinate_df(coordinate_df)


    def set_coordinate_df(self, coordinate_df):
        if coordinate_df is not None:
            self.coordinate_df = coordinate_df[coordinate_df.study_id == self.study_id]


    def get_largest_series(self, series_description):
        """
        Best used for initial segmentation, will return the largest continuous data
        """
        largest = None
        most_slices = 0
        for series_id, ssd, series in self.series:
            if series_description == ssd:
                stack = series.get_largest_stack()
                if stack.number_of_instances > most_slices:
                    most_slices = stack.number_of_instances
                    largest = series
        return largest
                    
    def __repr__(self):
        return f'OrientedStudy(study_id={self.study_id}, n_series={len(self.series)})'
        
