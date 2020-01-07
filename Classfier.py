# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 14:52:06 2019

@author: 김병철
"""
import os
import datetime
import pickle
import time
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv3D, MaxPooling3D, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import multi_gpu_model

from data_to_Array import to_Arrays_From_Dataset

# 학습에 사용한 이미지는 (32, 32, 32) 크기에 체널이 1개임
voxel = 32
dims = (voxel, voxel, voxel)
n_channels = 1
# 검증 데이터셋의 비율
validation_data_ratio = 0.2
# Epochs
num_epochs = 40
# Batch size
batch_size = 16
# 클래스 수
n_classes = 128
# 결과 저장 폴더
output_folder_path = "..\\Output"
# 데이터셋 폴더 : 실행 코드 상위 디렉토리에 두기
# (EX) dataset_dir = "..\\Dataset\\"
# 데이터 불러오기
load_exist_data = [False,""]
#%% GPU 개수를 알아낸다.
def get_available_gpus():
   local_device_protos = device_lib.list_local_devices()
   return [x.name for x in local_device_protos if x.device_type == 'GPU']
#%% 데이터셋을 준비한다. (셔플 포함)
def prepare_dataset():
    train_set, test_set = to_Arrays_From_Dataset(voxel)
    
    n_tr = train_set[0].shape[0]
    p_tr = list(np.random.permutation(n_tr))
    train_set[0] = train_set[0][p_tr]
    train_set[1] = train_set[1][p_tr]
    
    n_ts = test_set[0].shape[0]
    p_ts = list(np.random.permutation(n_ts))
    test_set[0] = test_set[0][p_ts]
    test_set[1] = test_set[1][p_ts]
    return train_set, test_set
#%% 데이터셋을 저장한다.
def save_dataset(train_set, test_set):
    now_time = datetime.datetime.now()
    folder_name = os.path.join(output_folder_path,
                               'Data_' + now_time.strftime('%Y%m%d_%H%M%S'))
    file_name = folder_name + '\\Data.p'
    if not os.path.isfile(file_name):
        os.makedirs(folder_name)
        with open(file_name, 'wb') as file:
            pickle.dump(train_set, file)
            pickle.dump(test_set, file)
#%% 데이터셋을 불러온다.
def load_dataset():
    with open(load_exist_data[1], 'rb') as file:
        train_set = pickle.load(file)
        test_set = pickle.load(file)
    return train_set, test_set
#%% 학습 그래프 출력
def plot_history(hist):
    plt.plot(hist.history['loss'], 'y', label='Train loss')
    plt.plot(hist.history['val_loss'], 'r', label='Val loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()
        
    plt.plot(hist.history['accuracy'], 'b', label='Train acc')
    plt.plot(hist.history['val_accuracy'], 'g', label='Val acc')    
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')    
    plt.legend(loc='lower right')
    plt.show()
#%% 모델 정의
def define_model():
    model = Sequential()    
        
    model.add(Conv3D(32, (5, 5, 5), input_shape=(*dims, n_channels), padding='same', activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    
    model.add(Conv3D(64, (5, 5, 5), padding='same', activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    
    model.add(Conv3D(128, (5, 5, 5), padding='same', activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    
    model.add(Flatten())
    model.add(Dense(units=n_classes, activation='softmax'))
        
    return model
#%% 학습 기록 저장
def save_history(hist):
    now_time = datetime.datetime.now()
    folder_name = os.path.join(output_folder_path,
                               'history_' + now_time.strftime('%Y%m%d_%H%M%S'))
    if not os.path.isfile(folder_name):
        os.makedirs(folder_name)
    with open(folder_name + '\\trainHistoryDict', 'wb') as file:
        pickle.dump(hist.history, file)
#%% 메인 함수
def main():
    start_time = datetime.datetime.now()            
        
    # 실행할 때마다 랜덤값에 영향을 받지 않게 시드를 통일시킨다.
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    random.seed(12345)
    tf.compat.v1.set_random_seed(1234)
        
    # GPU 사용 가능 확인
    num_gpu = len(get_available_gpus())
    use_gpu = num_gpu >= 2

    print("PlantEquipment Classifier")
    print(start_time)
    
    # Dataset 폴더에서 자동차 이미지와 비-자동차 이미지를 읽어온다.
    if load_exist_data[0] == True:
        train_set, test_set = load_dataset()
    else:
        train_set, test_set = prepare_dataset()
        save_dataset(train_set, test_set)
    
        
    m_train = train_set[0].shape[0] # 학습 데이터셋 개수
    m_test = test_set[0].shape[0] # 시험 데이터셋 개수   
        
    # 정보를 화면에 출력한다.
    print ("Number of GPUs: " + str(num_gpu))
    print ("Number of training examples: " + str(m_train))
    print ("Number of testing examples: " + str(m_test))
    print ("Number of epochs: " + str(num_epochs))
    print ("Batch size: " + str(batch_size))    
    print ("Image size: (" + str(dims[0]) + ", " + str(dims[1]) + ", " + str(dims[2]) + ")")
    print ("Input channel: " + str(n_channels))
    
    # 모델 생성
    if use_gpu:
        with tf.device('/cpu:0'):
            model = define_model()
            parallel_model = multi_gpu_model(model, gpus=num_gpu)
    else:
        model = define_model()
    
    # 모델 컴파일
    if use_gpu:
        parallel_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])        
    
    # output_folder_name이 없으면 폴더를 만든다.
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
            
    # 모델을 파일로 저장한다.
    with open(os.path.join(output_folder_path, 'model_architectures.json'), 'w') as f:
        f.write(model.to_json())
            
    # 콜벡 함수 지정
    
    callbacks = []    
        
    # 가장 좋은 모델을 저장
    checkpoint_best_path = output_folder_path + '\\model_best.h5'
    checkpoint_best = ModelCheckpoint(filepath=checkpoint_best_path, save_best_only=True, verbose=1)
    callbacks.append(checkpoint_best)
    
    # 모델 학습시키기
    print("Start Training")
    start_time = time.time()
    
    #history = parallel_model.fit(train_x, train_y, batch_size=batch_size, epochs=num_epochs)
    
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min') # 조기종료 콜백함수 정의
    callbacks.append(early_stopping) 
    
    if use_gpu:
        history = parallel_model.fit(train_set[0], train_set[1], batch_size = batch_size,
                                     epochs = num_epochs, validation_split = validation_data_ratio, callbacks=callbacks)
    else:
        history = model.fit(train_set[0], train_set[1], batch_size = batch_size,
                            epochs = num_epochs, validation_split = validation_data_ratio, callbacks=callbacks)
        
# =============================================================================
#     # 데이터 강화를 위한 ImageDataGenerator 생성
#     # 좌우 이동, 좌우 뒤집기, 확대/축소 적용
#     # 20%를 검증용으로 사용
#     datagen = ImageDataGenerator(width_shift_range=0.2,
#                                  height_shift_range=0.2, 
#                                  horizontal_flip=True,
#                                  zoom_range=0.2,
#                                  validation_split=0.2)
#     
#     # 데이터를 학습용과 검증용으로 분리
#     train_generator = datagen.flow(train_x, train_y, batch_size = batch_size, subset='training')
#     validation_generator = datagen.flow(train_x, train_y, batch_size = batch_size, subset='validation')
# 
#     # 모델 학습
#     if use_gpu:
#         history = parallel_model.fit_generator(train_generator,
#                                                steps_per_epoch=train_x.shape[0] / batch_size * 10,
#                                                epochs=num_epochs,
#                                                validation_data=validation_generator,
#                                                validation_steps=train_x.shape[0] / batch_size * 2)
#     else:
#         history = model.fit_generator(train_generator,
#                                                steps_per_epoch=train_x.shape[0] / batch_size * 10,
#                                                epochs=num_epochs,
#                                                validation_data=validation_generator,
#                                                validation_steps=train_x.shape[0] / batch_size * 2)
# =============================================================================
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("Training Finished!")
    print("Training Time(s): ", elapsed_time)    
    
    # 가중치를 저장한다.
    model.save_weights(os.path.join(output_folder_path, 'model_weights.h5'))
    # 학습 그래프 출력
    plot_history(history)
    save_history(history)
    
    # 시험셋을 이용해 모델 평가
    if use_gpu:
        score = parallel_model.evaluate(test_set[0], test_set[1])
    else:
        score = model.evaluate(test_set[0], test_set[1])
        
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
        
    return history
    
if __name__ == "__main__":
    hist = main()