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

# from keras.models import *
# from keras.layers import *
# from keras.optimizers import *
# from keras import backend as keras
# from keras.preprocessing.image import ImageDataGenerator
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler

# from glob import glob
# from tqdm import tqdm

# data_path = 'UNetJSRT/data/'
# label_path = 'UNetJSRT/label/heart/'
# dilate_path = 'UNetJSRT/dilate/heart/'
# data_type = '.png'
#
# file_list = return_list(data_path, data_type)
# test_img = file_list[1]
#
# image = cv2.imread(data_path+test_img)
# mask_image = cv2.imread(label_path+test_img)
# dilate_image = cv2.imread(dilate_path+test_img)
# merged_image = add_colored_dilate(image, mask_image, dilate_image)
#
# fig, axs = plt.subplots(1, 4, figsize=(15, 8))
#
# axs[0].set_title("X-Ray")
# axs[0].imshow(image)
#
# axs[1].set_title("Mask")
# axs[1].imshow(mask_image)
#
# axs[2].set_title("Dilate")
# axs[2].imshow(dilate_image)
#
# axs[3].set_title("Merged")
# axs[3].imshow(merged_image)
#
# plt.show()

test_data = '../UNetJSRT/val/IMG/'
test_mask = '../UNetJSRT/val/mask/'
input_size = 512
root_path = '../UNetJSRT/'

UNet_model = UNet_CUNet.DeepModel(size_set=input_size)
UNet_model.load_weights('../UNetJSRT/Model/UNet_Partial_FullModel_CUNet_Adam.h5')
# UNet_model = UNet.DeepModel(size_set=input_size)
# UNet_model.load_weights('./UNetJSRT/Model/UNetRLung_Adam.h5')
if(os.path.exists(os.path.join(root_path, 'test'))):
    shutil.rmtree(os.path.join(root_path, 'test'))
mk_dir(root_path+'test/')
#
test_files = return_list(test_data, '.png')
test_gen = test_generator(test_data, test_files, target_size=(512,512))
results = UNet_model.predict_generator(test_gen, len(test_files), verbose=1)

print(results[0].shape, results[0])
#

# arg_max = np.argmax(results[0], axis = 2)
# # print('max shape', arg_max.shape, arg_max)
# img = np.zeros((512,512,1))
# # img[arg_max==0, 0] =1
# # img[arg_max==1, 0] =1
# img[arg_max==2, 0] =1
# print('img shape', img.shape)
# for i in range(len(results)):
#     arg_max = np.argmax(results[i], axis = 2)
#     # print('max shape', arg_max.shape, arg_max)
#     img = np.zeros((512,512,1))
#     # img[arg_max==0, 0] =1
#     # img[arg_max==1, 0] =1
#     img[arg_max==2, 0] =1
#     results[i]  = img
# print('test file 0', test_files[0])

# save_result(root_path+'test/', results, test_files)

# save_result_multiclass(root_path+'test/', results, test_files, 3)
save_result_multilabel(root_path+'test/', results, test_files, 6)
#

# image = cv2.imread(test_data+test_files[19])
# mask_image = cv2.imread(test_mask+test_files[19])
# predict_mask_image = cv2.imread(root_path+'test/'+test_files[19])
# original_image = add_colored_mask(image, mask_image)
# predicted_image = add_colored_mask(image, predict_mask_image)
#
# fig, axs = plt.subplots(1, 4, figsize=(15, 8))
#
# axs[0].set_title("CT original")
# axs[0].imshow(image)
#
# axs[1].set_title("Ground truth")
# axs[1].imshow(original_image)
#
# axs[2].set_title("Predicted result")
# axs[2].imshow(predicted_image)
#
# axs[3].set_title("diff")
# axs[3].imshow(diff_mask(mask_image, predict_mask_image))
# fig.suptitle(test_files[19], fontsize=16)
# plt.show()
