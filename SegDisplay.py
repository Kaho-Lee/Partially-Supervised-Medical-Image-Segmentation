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

save_result_multilabel(root_path+'test/', results, test_files, 6)

