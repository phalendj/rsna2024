import numpy as np

import datasets.dicom_load as dcl


def test_oriented_stack1():
    study_id = 4003253
    series_id = 2448190387  
    series_description = 'Axial T2'
    path_to_dicom = dcl.get_series_directory(study_id, series_id)

    series = dcl.OrientedSeries(path_to_dicom=path_to_dicom, series_description=series_description)
    series.load()
    assert series.study_id == study_id
    assert series.series_id == series_id
    assert series.series_description == series_description
    assert len(series.dicom_stacks) == 2
    stack = series.dicom_stacks[1]
    assert stack.data.shape == (23, 320, 320)

    s = np.stack([i*np.ones((320,320)) for i in range(23)])
    stack.dicom_info['array'] = s

    t, instance_numbers = stack.get_thick_slice(instance_number=7, slice_thickness=5, boundary_instance=None, center=False)
    assert np.all(t[:, 0, 0] == np.array([4,5,6,7,8]))
    assert np.all(instance_numbers == np.array([5, 6,7, 8, 9]))

    t, instance_numbers = stack.get_thick_slice(instance_number=1, slice_thickness=5, boundary_instance=None, center=False)
    assert np.all(t[:, 0, 0] == np.array([0,1,2,3,4]))
    assert np.all(instance_numbers == np.array([1, 2,3, 4, 5]))

    t, instance_numbers = stack.get_thick_slice(instance_number=22, slice_thickness=5, boundary_instance=None, center=False)
    assert np.all(t[:, 0, 0] == np.array([18,19,20,21,22]))

    t, instance_numbers = stack.get_thick_slice(instance_number=1, slice_thickness=5, boundary_instance=None, center=True)
    assert np.all(t[:, 0, 0] == np.array([0,1,2,0,0]))

    t, instance_numbers = stack.get_thick_slice(instance_number=23, slice_thickness=5, boundary_instance=None, center=True)
    assert np.all(t[:, 0, 0] == np.array([20,21,22,0,0]))

    t, instance_numbers = stack.get_thick_slice(instance_number=2, slice_thickness=5, boundary_instance=None, center=True)
    assert np.all(t[:, 0, 0] == np.array([0,1,2,3,0]))

    t, instance_numbers = stack.get_thick_slice(instance_number=22, slice_thickness=5, boundary_instance=None, center=True)
    assert np.all(t[:, 0, 0] == np.array([19,20,21,22,0]))
    assert np.all(instance_numbers == np.array([20,21,22,23,-1]))

    t, instance_numbers = stack.get_thick_slice(instance_number=22, slice_thickness=30, boundary_instance=None, center=False)
    assert np.all(t[:, 0, 0] == np.array(list(range(23)) +[0]*7))


    t, instance_numbers = stack.get_thick_slice(instance_number=22, slice_thickness=30, boundary_instance=None, center=True)
    assert np.all(t[:, 0, 0] == np.array(list(range(6,23)) +[0]*13))

    t, instance_numbers = stack.get_thick_slice(instance_number=7, slice_thickness=5, boundary_instance=6, center=False)
    assert np.all(t[:, 0, 0] == np.array([6,7,8, 9, 10]))

    t, instance_numbers = stack.get_thick_slice(instance_number=7, slice_thickness=5, boundary_instance=9, center=False)
    assert np.all(t[:, 0, 0] == np.array([3,4,5,6, 7]))

    t, instance_numbers = stack.get_thick_slice(instance_number=7, slice_thickness=5, boundary_instance=9, center=True)
    assert np.all(t[:, 0, 0] == np.array([4,5,6, 7, 0]))

    t, instance_numbers = stack.get_thick_slice(instance_number=7, slice_thickness=5, boundary_instance=5, center=True)
    assert np.all(t[:, 0, 0] == np.array([0,5,6, 7, 8]))
    assert np.all(instance_numbers == np.array([-1, 6,7, 8, 9]))

    instance_number = 3
    x = 179.12
    y = 161.23
    wx, wy, wz = stack.get_world_coordinates(instance_number=instance_number, x=x, y=y)
    assert np.all(np.round(np.array([wx, wy, wz]), 6) == np.array([16.690849, 74.619268, -386.373537]))
    z0, x0, y0 = stack.get_pixel_from_world(wx, wy, wz)
    assert (z0, x0, y0) == (2, 179, 161)

def test_oriented_stack2():
    study_id = 4003253
    series_id = 2448190387  
    series_description = 'Axial T2'
    path_to_dicom = dcl.get_series_directory(study_id, series_id)

    series = dcl.OrientedSeries(path_to_dicom=path_to_dicom, series_description=series_description)
    series.load()
    stack = series.dicom_stacks[1]
    assert stack.data.shape == (23, 320, 320)

    s = np.stack([i*np.ones((320,320)) for i in range(23)])
    for i in range(320):
        s[:, i, :] += 0.001*i
    for j in range(320):
        s[:, :, j] += 0.000001*j
    stack.dicom_info['array'] = s

    t, instance_numbers = stack.get_thick_patch(instance_number=7, slice_thickness=5, x=317, y=317, patch_size=20, boundary_instance=None, center=False, center_patch=False)
    assert t[0,0,0] == 4.3003
    assert t[0,-1,-1] == 4.319319
    assert np.all(instance_numbers == np.array([5, 6, 7, 8, 9]))

    t, instance_numbers = stack.get_thick_patch(instance_number=7, slice_thickness=5, x=5, y=5, patch_size=20, boundary_instance=None, center=False, center_patch=False)
    assert t[0,0,0] == 4.0
    assert t[0,-1,-1] == 4.019019

    t, instance_numbers = stack.get_thick_patch(instance_number=7, slice_thickness=5, x=5, y=5, patch_size=20, boundary_instance=None, center=False, center_patch=True)
    assert t[0,0,0] == 0.0
    assert t[0,5,5] == 4.0
    assert t[0,-1,-1] == 4.014014

    t, instance_numbers = stack.get_thick_patch(instance_number=7, slice_thickness=5, x=315, y=315, patch_size=20, boundary_instance=None, center=False, center_patch=True)
    assert t[0,0,0] == 4.305305
    assert t[0,-6,-6] == 4.319319
    assert t[0,-1,-1] == 0

    t, instance_numbers = stack.get_thick_patch(instance_number=7, slice_thickness=5, x=115, y=115, patch_size=20, boundary_instance=None, center=False, center_patch=False)

    assert t[0,0,0] == 4.105105
    assert np.round(t[0,-1,-1],6) == 4.124124

    t, instance_numbers = stack.get_thick_patch(instance_number=7, slice_thickness=5, x=115, y=115, patch_size=20, boundary_instance=None, center=False, center_patch=True)

    assert t[0,0,0] == 4.105105
    assert np.round(t[0,-1,-1],6) == 4.124124
