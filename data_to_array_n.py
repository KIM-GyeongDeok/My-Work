# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 16:45:25 2020

@author: 김병철
"""
import vtk
from vtk.util import numpy_support

def load_voxel_file_from_mha(full_path):
    reader = vtk.vtkMetaImageReader()
    reader.SetFileName(full_path)
    reader.Update()

    image_data = reader.GetOutput()
    dims = image_data.GetDimensions()
    
    np_data = numpy_support.vtk_to_numpy(image_data.GetPointData().GetScalars())
    np_data = np_data.reshape(*dims, order='C')

    return np_data
