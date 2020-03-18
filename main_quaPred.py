import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
from Utils import *
from Utils_Quality import *
import UNet
import UNet_CUNet
import shutil
import UNet_MultiStructure
import os
import UNetDropOut
import VGG

import random
from keras.optimizers import SGD
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf


# os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)

BATCH_SIZE=1

EPOCHS=10

test_data = '../UNetJSRT/val/IMG/'
test_mask = '../UNetJSRT/val/mask/'
train_data = '../UNetJSRT/train/IMG/'
train_mask = '../UNetJSRT/train/mask/'

input_size = 512
root_path = '../UNetJSRT/'

# UNet_model = UNetDropOut.DeepModel(size_set=input_size)
# UNet_model.load_weights('../UNetJSRT/Model/UNet_Partial_UNetDropOut_sig_Adam.h5')
# # UNet_model = UNet.DeepModel(size_set=input_size)
# # UNet_model.load_weights('./UNetJSRT/Model/UNetRLung_Adam.h5')
# if(os.path.exists(os.path.join(root_path, 'qualityPredVal'))):
#     shutil.rmtree(os.path.join(root_path, 'qualityPredVal'))
# mk_dir(root_path+'qualityPredVal/')
# #
# test_files = return_list(test_data, '.png')
# test_gen = test_generator(test_data, test_files, target_size=(512,512))
# results = UNet_model.predict_generator(test_gen, len(test_files), verbose=1)
#
# save_result_multilabel(root_path+'qualityPredVal/', results, test_files, 6)
# maxProb(root_path+'qualityPredVal/'+'MaxUncertainty/', results, test_files)
# MCDropOut(UNet_model, root_path+'qualityPredVal/'+'MCUncertainty/', results, test_data, test_files)
#
#
# if(os.path.exists(os.path.join(root_path, 'qualityPredTrain'))):
#     shutil.rmtree(os.path.join(root_path, 'qualityPredTrain'))
# mk_dir(root_path+'qualityPredTrain/')
# #
# train_files = return_list(train_data, '.png')
# train_gen = test_generator(train_data, train_files, target_size=(512,512))
# results = UNet_model.predict_generator(train_gen, len(train_files), verbose=1)
#
# save_result_multilabel(root_path+'qualityPredTrain/', results, train_files, 6)
# maxProb(root_path+'qualityPredTrain/'+'MaxUncertainty/', results, train_files)
# MCDropOut(UNet_model, root_path+'qualityPredTrain/'+'MCUncertainty/', results, train_data, train_files)



BATCH_SIZE=1

EPOCHS=10
EPISODES = 1
train_path = '../UNetJSRT/qualityPredTest/'
val_path = '../UNetJSRT/qualityPredVal/'
ImgPath = '../UNetJSRT/'
qualityImgPathTrain = '../UNetJSRT/qualityPredTrain/'
qualityImgPathVal = '../UNetJSRT/qualityPredVal/'
data_type = '.png'
save_model_file = '../UNetJSRT/Model/VGG_MaxProb_Eva.h5'

train_list = return_list(qualityImgPathTrain+'class1/', data_type)
print('len train', len(train_list))
val_list = return_list(qualityImgPathVal+'class1/', data_type)
print('len val', len(val_list))


for j in range(EPISODES):
    input_size = 224
    Optimizer_Adam = Adam(lr=5e-5)

    VGG_model = VGG.VGG_16qua(size_set=input_size)

    VGG_model.compile(optimizer=Optimizer_Adam, loss=RMSE_Jacc, metrics=[RMSE_Jacc])
    VGG_model.summary()

    train_gen = QualityEvaGenerator_Train(BATCH_SIZE,
                                ImgPath+'train', 'IMG','right_lung','left_lung',
                                'heart','left_clavicle','right_clavicle',
                                qualityImgPathTrain, 'MaxUncertainty','class1', 'class2',
                                'class3', 'class4', 'class5',
                                target_size=(512,512))

    val_gen = QualityEvaGenerator_Train(BATCH_SIZE,
                                ImgPath+'val', 'IMG','right_lung','left_lung',
                                'heart','left_clavicle','right_clavicle',
                                qualityImgPathVal, 'MaxUncertainty','class1', 'class2',
                                'class3', 'class4', 'class5',
                                target_size=(512,512))

    history = VGG_model.fit_generator(train_gen,
                                  steps_per_epoch=len(train_list) / BATCH_SIZE,
                                  epochs=EPOCHS,
                                  shuffle=True,
                                  validation_data = val_gen,
                                  validation_steps=len(val_list))

    VGG_model1.save(save_model_file)
