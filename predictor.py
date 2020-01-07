# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 13:22:13 2020

@author: 김병철
"""
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from data_to_Array import to_Array_From_File

def predict_From_File(file_DIR, model_DIR = "..\\Output\\model_best.h5", cls_dict_DIR ="Class_Dict"):
    class_dict = {}
    with open(cls_dict_DIR, 'rb') as file:
        class_dict = pickle.load(file)
        
    x = to_Array_From_File(file_DIR)
    x = np.array([x])
    model = load_model(model_DIR)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    y = model.predict(tf.cast(x, tf.float32))
    y = np.reshape(y, y.shape[1])
    y = list(y)
    return class_dict[y.index(max(y))] 

# =============================================================================
# file_DIR = "testdata\\1b40357d-7e14-5998-84ca-d69a64157f22.mhd"
# result_predicted = predict_From_File(file_DIR)
# from visualization import showImange_From_File
# showImange_From_File(file_DIR, result_predicted)
# =============================================================================
