#
import numpy as np
import scipy.io as sio
# from keras.preprocessing import image
from skimage.transform import rotate, resize
from skimage.measure import label, regionprops
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from Uttils import *
import UNet as UNet
import os
import Model_DiscSeg as DiscModel

structure_list = ['heart/', 'right_lung/', 'left_lung/', 'right_clavicle/', 'left_clavicle/']
shape_in = (2048, 2048) # matrix size
shape_mask = (1024,1024)
new_shape = (512,512)
dtype = np.dtype('>u2')

# disc_list = [400, 500, 600, 700, 800]
# DiscROI_size = 800
# DiscSeg_size =
# CDRSeg_size = 400

data_type = '.gif'
target_type = '.png'
# data_img_path = '../data/REFUGE-Training400/Training400/Glaucoma/'
data_img_path = 'JSRT/All247/'
label_img_path = 'JSRT/scratch/All247imagesMask/'
DilateKernel = np.ones((15, 15), np.uint8)

# data_save_path = mk_dir('../training_crop/data/')
# label_save_path = mk_dir('../training_crop/label/')
data_save_path = mk_dir('UNetJSRT/data/')
label_save_path = mk_dir('UNetJSRT/label/')
dilate_save_path = mk_dir('UNetJSRT/dilate/')


file_test_list = return_list(data_img_path, target_type)
for lineIdx in range(len(file_test_list)):
    temp_txt = file_test_list[lineIdx]
    print(' Resizing Img training image' + str(lineIdx + 1) + ': ' + temp_txt)
    image = cv2.imread(data_img_path+temp_txt)
    image = cv2.resize(image, new_shape)
    cv2.imwrite(data_save_path+temp_txt, image)


# DiscSeg_model = UNet.DeepModel(size_set=DiscSeg_size)
# DiscSeg_model.load_weights('./deep_model/Model_DiscSeg_ORIGA.h5')
for item in structure_list:
    structure_path = os.path.join(label_img_path, item)
    print(structure_path)
    # file_test_list = return_list(structure_path, target_type)

    if(not os.path.exists(os.path.join(label_save_path, item))):
        mk_dir(label_save_path+item)
    if(not os.path.exists(os.path.join(dilate_save_path, item))):
        mk_dir(dilate_save_path+item)

    """
       preprocessing data label
    """

    file_test_list = return_list(structure_path, target_type)
    for lineIdx in range((len(file_test_list))):
        temp_txt = file_test_list[lineIdx]
        # print(' Resizing Img ' + str(lineIdx + 1) + ': ' + temp_txt)
        mask = cv2.imread(structure_path+temp_txt)
        mask = cv2.resize(mask, new_shape)
        cv2.imwrite(label_save_path+item+temp_txt, mask)
        mask_dilate = cv2.dilate(mask, DilateKernel, iterations=1)
        cv2.imwrite(dilate_save_path+item+temp_txt, mask_dilate)

    """
       converting .gif to .png
    """
    # for lineIdx in range(len(file_test_list)):
    #
    #     temp_txt = file_test_list[lineIdx]
    #     temp_name = temp_txt[:-4]
    #     print(' Processing Img ' + str(lineIdx + 1) + ': ' + temp_txt + ' ' +temp_name)
    #
    #     im = Image.open(structure_path+temp_txt)
    #     bg = Image.new("L", im.size)
    #     bg.paste(im, (0,0), im)
    #     bg.save(structure_path+temp_name+target_type, quality=95)









        # load image
        # org_img = np.asarray(image.load_img(data_img_path + temp_txt))


        # load label
        # org_label = np.asarray(image.load_img(label_img_path + temp_txt[:-4] + '.bmp'))[:,:,0]
        # new_label = np.zeros(np.shape(org_label) + (3,), dtype=np.uint8)
        # new_label[org_label < 200, 0] = 255
        # new_label[org_label < 100, 1] = 255
