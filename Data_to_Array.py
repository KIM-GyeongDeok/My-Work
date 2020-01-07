import os
import numpy as np
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from medpy.io import load


def to_Arrays_From_Dataset(voxel, dataset_dir = os.path.abspath(os.path.join(os.curdir, os.pardir)) + "\\Dataset\\"):
    dataset_dir = dataset_dir + "Voxel_%s\\"%voxel
    class_list = array(os.listdir(dataset_dir+"Training\\"))
    
    label_encoder = LabelEncoder()  
    integer_encoded = label_encoder.fit_transform(class_list)
    onehot_encoder = OneHotEncoder(sparse=False) 
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    
    train_test=['Training\\','Test\\']
    train_set = [0,0]
    test_set = [0,0]
    for train_or_test in train_test:
        input_array = []
        label_array = []
        path = dataset_dir + train_or_test
        for classindex, classname in enumerate(class_list):
            class_path = path + classname + "\\"
            file_list = os.listdir(class_path)
            for filename in file_list:
                if(filename[-4:] != "zraw" ):
                    file_path = class_path + filename
                    data, header  = load(file_path)
                    input_array.append(np.array(data))
                    label_array.append(np.array(onehot_encoded[classindex]))
        input_array = np.reshape(input_array, (np.shape(input_array) + (1,)))
        label_array = np.array(label_array)
        if train_or_test =='Training\\':
            train_set[0]=input_array
            train_set[1]=label_array
        else:
            test_set[0]=input_array
            test_set[1]=label_array
        
    return train_set, test_set

def to_Array_From_File(DIR):
    data, header  = load(DIR)
    return np.reshape(data, (np.shape(data) + (1,)))
    

# =============================================================================
# train, test = to_Arrays_From_Dataset(32)
# train_x = train[0]
# train_y = train[1]
# for i_obj, obj in enumerate(train[0]):
#     for i_x, x in enumerate(obj):
#         for i_y, y in enumerate(x):
#             for i_z, z in enumerate(y):
#                 if train_x[i_obj][i_x,i_y,i_z,0]!=z:
#                     print("fail")
# =============================================================================
                
        