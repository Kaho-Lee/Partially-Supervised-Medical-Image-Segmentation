import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
from Utils import *
from mnet_utils import BW_img, disc_crop, mk_dir, return_list
import UNet
import UNet_CUNet
import shutil
import UNet_MultiStructure
import os
import UNetDropOut

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

UNet_model = UNetDropOut.DeepModel(size_set=input_size)
UNet_model.load_weights('../UNetJSRT/Model/UNet_Partial_UNetDropOut_sig_Adam.h5')
# UNet_model = UNet.DeepModel(size_set=input_size)
# UNet_model.load_weights('./UNetJSRT/Model/UNetRLung_Adam.h5')
if(os.path.exists(os.path.join(root_path, 'qualityPred'))):
    shutil.rmtree(os.path.join(root_path, 'qualityPred'))
mk_dir(root_path+'qualityPred/')
#
test_files = return_list(test_data, '.png')
test_gen = test_generator(test_data, test_files, target_size=(512,512))
results = UNet_model.predict_generator(test_gen, len(test_files), verbose=1)

print(results[0].shape, results[0])
#


# save_result(root_path+'test/', results, test_files)

# save_result_multiclass(root_path+'test/', results, test_files, 3)
save_result_multilabel(root_path+'qualityPred/', results, test_files, 6)
maxProb(root_path+'qualityPred/'+'uncertainty/', results, test_files)
MCDropOut(UNet_model, root_path+'qualityPred/'+'MCUncertainty/', results, test_files)
