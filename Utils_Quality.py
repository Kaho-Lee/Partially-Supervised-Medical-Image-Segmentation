"""
    Taken from https://www.kaggle.com/eduardomineo/u-net-lung-segmentation-montgomery-shenzhen/data on 0309,2019
"""
from __future__ import print_function
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import keras.backend as K
from keras import utils as np_utils
from Utils import *

from glob import glob
from tqdm import tqdm

'''
Segmentation Uncertainty
'''


def maxProb(save_path, npyfile, test_files):
    if(os.path.exists(save_path)):
        shutil.rmtree(save_path)
    mk_dir(save_path)

    for i, item in enumerate(npyfile):
        result_file = test_files[i]
        filename, fileext = os.path.splitext(os.path.basename(result_file))
        maxUncertainty = (np.max(item, axis=2) * 255.).astype(np.uint8)
        result_file = os.path.join(save_path, "%s%s" % (filename, fileext))
        cv2.imwrite(result_file, maxUncertainty)

def MCDropOut(model, save_path, npyfile, test_files_path, test_files):
    if(os.path.exists(save_path)):
        shutil.rmtree(save_path)
    mk_dir(save_path)
    # test_files_path = 'UNetJSRT/val/IMG/'
    # test_gen = test_generator(test_data, test_files, target_size=(512,512))
    # results = UNet_model.predict_generator(test_gen, len(test_files), verbose=1)

    T = 20
    total = npyfile
    for i in range(T-1):
        print('in pediction ', i+2)
        test_gen = test_generator(test_files_path, test_files, target_size=(512,512))
        results = model.predict_generator(test_gen, len(test_files), verbose=1)
        total = np.add(total, results)

    total = np.divide(total, T)

    for i, item in enumerate(total):
        result_file = test_files[i]
        filename, fileext = os.path.splitext(os.path.basename(result_file))
        uncertainty =  np.multiply(-1, np.multiply(item, np.log(item+0.0000000001)))
        uncertainty = np.sum(uncertainty, axis=2)
        mcUncertainty = ((1-uncertainty) * 255.).astype(np.uint8)
        result_file = os.path.join(save_path, "%s%s" % (filename, fileext))
        cv2.imwrite(result_file, mcUncertainty)


'''
End of uncertainty measurement
'''

def jaccard_index(ground_truth, predicted_mask):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.

    Ref: https://en.wikipedia.org/wiki/Jaccard_index

    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """

    intersection = np.sum(np.abs(np.multiply(ground_truth, predicted_mask)))
    sum_ = np.sum(np.abs(ground_truth) + np.abs(predicted_mask))
    jac = intersection / (sum_ - intersection)

    return jac

def RMSE_Jacc(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def QualityEvaGenerator_Train(batch_size, aug_dict, ground_truth_path, image_folder,
        mask_right_lung_folder, mask_left_lung_folder, mask_heart_folder,
        mask_left_clavicle_folder, mask_right_clavicle_folder,
        pred_path, pred_uncertainty,
        mask_right_lung_pred_folder, mask_left_lung_pred_folder, mask_heart_pred_folder,
        mask_left_clavicle_pred_folder, mask_right_clavicle_pred_folder,
        image_color_mode="grayscale",
        mask_color_mode="grayscale",
        image_save_prefix="image",
        mask_save_prefix="mask",
        save_to_dir=None,
        target_size=(256,256),
        seed=1):

        '''
        can generate image and mask at the same time use the same seed for
        image_datagen and mask_datagen to ensure the transformation for image
        and mask is the same if you want to visualize the results of generator,
        set save_to_dir = "your path"
        '''
        image_datagen = ImageDataGenerator()

        mask_datagen_right_lung = ImageDataGenerator()
        mask_datagen_left_lung = ImageDataGenerator()
        mask_datagen_heart = ImageDataGenerator()
        mask_datagen_left_clavicle  = ImageDataGenerator()
        mask_datagen_right_clavicle = ImageDataGenerator()

        uncertainty_datagen = ImageDataGenerator()

        mask_datagen_right_lung_pred = ImageDataGenerator()
        mask_datagen_left_lung_pred = ImageDataGenerator()
        mask_datagen_heart_pred = ImageDataGenerator()
        mask_datagen_left_clavicle_pred  = ImageDataGenerator()
        mask_datagen_right_clavicle_pred = ImageDataGenerator()



        print('111111')

        image_generator = image_datagen.flow_from_directory(
            ground_truth_path,
            classes = [image_folder],
            class_mode = None,
            color_mode = image_color_mode,
            target_size = target_size,
            batch_size = batch_size,
            save_to_dir = save_to_dir,
            save_prefix  = image_save_prefix,
            seed = seed)

        mask_generator_right_lung = mask_datagen_right_lung.flow_from_directory(
            ground_truth_path,
            classes = [mask_right_lung_folder],
            class_mode = None,
            color_mode = mask_color_mode,
            target_size = target_size,
            batch_size = batch_size,
            save_to_dir = save_to_dir,
            save_prefix  = mask_save_prefix,
            seed = seed)

        mask_generator_left_lung = mask_datagen_left_lung.flow_from_directory(
            ground_truth_path,
            classes = [mask_left_lung_folder],
            class_mode = None,
            color_mode = mask_color_mode,
            target_size = target_size,
            batch_size = batch_size,
            save_to_dir = save_to_dir,
            save_prefix  = mask_save_prefix,
            seed = seed)

        mask_generator_heart = mask_datagen_heart.flow_from_directory(
            ground_truth_path,
            classes = [mask_heart_folder],
            class_mode = None,
            color_mode = mask_color_mode,
            target_size = target_size,
            batch_size = batch_size,
            save_to_dir = save_to_dir,
            save_prefix  = mask_save_prefix,
            seed = seed)

        mask_generator_left_clavicle = mask_datagen_left_clavicle.flow_from_directory(
            ground_truth_path,
            classes = [mask_left_clavicle_folder],
            class_mode = None,
            color_mode = mask_color_mode,
            target_size = target_size,
            batch_size = batch_size,
            save_to_dir = save_to_dir,
            save_prefix  = mask_save_prefix,
            seed = seed)

        mask_generator_right_clavicle = mask_datagen_right_clavicle.flow_from_directory(
            ground_truth_path,
            classes = [mask_right_clavicle_folder],
            class_mode = None,
            color_mode = mask_color_mode,
            target_size = target_size,
            batch_size = batch_size,
            save_to_dir = save_to_dir,
            save_prefix  = mask_save_prefix,
            seed = seed)

        '''
        pred
        '''
        uncertainty_generator = uncertainty_datagen.flow_from_directory(
            pred_path,
            classes = [pred_uncertainty],
            class_mode = None,
            color_mode = image_color_mode,
            target_size = target_size,
            batch_size = batch_size,
            save_to_dir = save_to_dir,
            save_prefix  = image_save_prefix,
            seed = seed)

        mask_generator_right_lung_pred = mask_datagen_right_lung_pred.flow_from_directory(
            pred_path,
            classes = [mask_right_lung_pred_folder],
            class_mode = None,
            color_mode = mask_color_mode,
            target_size = target_size,
            batch_size = batch_size,
            save_to_dir = save_to_dir,
            save_prefix  = mask_save_prefix,
            seed = seed)

        mask_generator_left_lung_pred = mask_datagen_left_lung_pred.flow_from_directory(
            pred_path,
            classes = [mask_left_lung_pred_folder],
            class_mode = None,
            color_mode = mask_color_mode,
            target_size = target_size,
            batch_size = batch_size,
            save_to_dir = save_to_dir,
            save_prefix  = mask_save_prefix,
            seed = seed)

        mask_generator_heart_pred = mask_datagen_heart_pred.flow_from_directory(
            pred_path,
            classes = [mask_heart_pred_folder],
            class_mode = None,
            color_mode = mask_color_mode,
            target_size = target_size,
            batch_size = batch_size,
            save_to_dir = save_to_dir,
            save_prefix  = mask_save_prefix,
            seed = seed)

        mask_generator_left_clavicle_pred = mask_datagen_left_clavicle_pred.flow_from_directory(
            pred_path,
            classes = [mask_left_clavicle_pred_folder],
            class_mode = None,
            color_mode = mask_color_mode,
            target_size = target_size,
            batch_size = batch_size,
            save_to_dir = save_to_dir,
            save_prefix  = mask_save_prefix,
            seed = seed)

        mask_generator_right_clavicle_pred = mask_datagen_right_clavicle_pred.flow_from_directory(
            pred_path,
            classes = [mask_right_clavicle_pred_folder],
            class_mode = None,
            color_mode = mask_color_mode,
            target_size = target_size,
            batch_size = batch_size,
            save_to_dir = save_to_dir,
            save_prefix  = mask_save_prefix,
            seed = seed)


        train_gen = zip(image_generator, mask_generator_right_lung, mask_generator_left_lung, mask_generator_heart,
                            mask_generator_left_clavicle, mask_generator_right_clavicle,
                        uncertainty_generator, mask_generator_right_lung_pred, mask_generator_left_lung_pred, mask_generator_heart_pred,
                            mask_generator_left_clavicle_pred, mask_generator_right_clavicle_pred)

        for (img, mask_right_lung, mask_left_lung, mask_heart, mask_left_clavicle, mask_right_clavicle,
            uncertainty, mask_right_lung_pred, mask_left_lung_pred, mask_heart_pred, mask_left_clavicle_pred, mask_right_clavicle_pred) in train_gen:

            img, mask_right_lung, mask_left_lung, mask_heart, mask_left_clavicle, mask_right_clavicle = resize_data_multi(img, mask_right_lung, mask_left_lung, mask_heart, mask_left_clavicle, mask_right_clavicle, 224)
            uncertainty, mask_right_lung_pred, mask_left_lung_pred, mask_heart_pred, mask_left_clavicle_pred, mask_right_clavicle_pred = resize_data_multi(uncertainty, mask_right_lung_pred, mask_left_lung_pred, mask_heart_pred, mask_left_clavicle_pred, mask_right_clavicle_pred, 224)
            # print('adjust img shape ', img.shape)
            img = img[:,:,:,0]
            uncertainty = uncertainty[:,:,:,0]

            mask_right_lung = mask_right_lung[:, :, :, 0]
            mask_left_lung = mask_left_lung[:, :, :, 0]
            mask_heart = mask_heart[:, :, :, 0]
            mask_left_clavicle = mask_left_clavicle[:, :, :, 0]
            mask_right_clavicle = mask_right_clavicle[:, :, :, 0]

            mask_right_lung_pred = mask_right_lung_pred[:,:,:,0]
            mask_left_lung_pred = mask_left_lung_pred[:,:,:,0]
            mask_heart_pred = mask_heart_pred[:,:,:,0]
            mask_left_clavicle_pred = mask_left_clavicle_pred[:,:,:,0]
            mask_right_clavicle_pred = mask_right_clavicle_pred[:,:,:,0]

            mask = np.zeros(mask_heart.shape + (7,)) #5 class (except background), 1 img, 1 uncertainty
            quality_value = np.zeros((1,1))
            # print('img shape ', img.shape)
            # print('uncertainty shape ', uncertainty.shape)
            # print('mask shape 0 ', mask[:, :, :, 0].shape)
            # print('mask shape ', mask.shape)
            mask[:, :, :, 0] = img
            mask[:, :, :, 1] = uncertainty
            mask[mask_right_lung_pred==1, 2] = 1
            quality_value = np.add(quality_value, jaccard_index(mask_right_lung, mask_right_lung_pred))
            mask[mask_left_lung_pred==1, 3] = 1
            quality_value = np.add(quality_value, jaccard_index(mask_left_lung, mask_left_lung_pred))
            mask[mask_heart_pred==1, 4] = 1
            quality_value = np.add(quality_value, jaccard_index(mask_heart, mask_heart_pred))
            mask[mask_left_clavicle_pred==1, 5] = 1
            quality_value = np.add(quality_value, jaccard_index(mask_left_clavicle, mask_left_clavicle_pred))
            mask[mask_right_clavicle_pred==1, 6] =1
            quality_value = np.add(quality_value, jaccard_index(mask_right_clavicle, mask_right_clavicle_pred))


            yield(mask, np.divide(quality_value, 5))


def QualityEvaGenerator_Test(batch_size, ground_truth_path, image_folder,
                              mask_right_lung_folder, mask_left_lung_folder, mask_heart_folder,
                              mask_left_clavicle_folder, mask_right_clavicle_folder,
                              pred_path, pred_uncertainty,
                              mask_right_lung_pred_folder, mask_left_lung_pred_folder, mask_heart_pred_folder,
                              mask_left_clavicle_pred_folder, mask_right_clavicle_pred_folder,
                              image_color_mode="grayscale",
                              mask_color_mode="grayscale",
                              image_save_prefix="image",
                              mask_save_prefix="mask",
                              save_to_dir=None,
                              target_size=(256, 256),
                              seed=1):
    '''
    can generate image and mask at the same time use the same seed for
    image_datagen and mask_datagen to ensure the transformation for image
    and mask is the same if you want to visualize the results of generator,
    set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator()

    mask_datagen_right_lung = ImageDataGenerator()
    mask_datagen_left_lung = ImageDataGenerator()
    mask_datagen_heart = ImageDataGenerator()
    mask_datagen_left_clavicle = ImageDataGenerator()
    mask_datagen_right_clavicle = ImageDataGenerator()

    uncertainty_datagen = ImageDataGenerator()

    mask_datagen_right_lung_pred = ImageDataGenerator()
    mask_datagen_left_lung_pred = ImageDataGenerator()
    mask_datagen_heart_pred = ImageDataGenerator()
    mask_datagen_left_clavicle_pred = ImageDataGenerator()
    mask_datagen_right_clavicle_pred = ImageDataGenerator()

    print('111111')

    image_generator = image_datagen.flow_from_directory(
        ground_truth_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)

    mask_generator_right_lung = mask_datagen_right_lung.flow_from_directory(
        ground_truth_path,
        classes=[mask_right_lung_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)

    mask_generator_left_lung = mask_datagen_left_lung.flow_from_directory(
        ground_truth_path,
        classes=[mask_left_lung_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)

    mask_generator_heart = mask_datagen_heart.flow_from_directory(
        ground_truth_path,
        classes=[mask_heart_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)

    mask_generator_left_clavicle = mask_datagen_left_clavicle.flow_from_directory(
        ground_truth_path,
        classes=[mask_left_clavicle_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)

    mask_generator_right_clavicle = mask_datagen_right_clavicle.flow_from_directory(
        ground_truth_path,
        classes=[mask_right_clavicle_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)

    '''
    pred
    '''
    uncertainty_generator = uncertainty_datagen.flow_from_directory(
        pred_path,
        classes=[pred_uncertainty],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)

    mask_generator_right_lung_pred = mask_datagen_right_lung_pred.flow_from_directory(
        pred_path,
        classes=[mask_right_lung_pred_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)

    mask_generator_left_lung_pred = mask_datagen_left_lung_pred.flow_from_directory(
        pred_path,
        classes=[mask_left_lung_pred_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)

    mask_generator_heart_pred = mask_datagen_heart_pred.flow_from_directory(
        pred_path,
        classes=[mask_heart_pred_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)

    mask_generator_left_clavicle_pred = mask_datagen_left_clavicle_pred.flow_from_directory(
        pred_path,
        classes=[mask_left_clavicle_pred_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)

    mask_generator_right_clavicle_pred = mask_datagen_right_clavicle_pred.flow_from_directory(
        pred_path,
        classes=[mask_right_clavicle_pred_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)

    train_gen = zip(image_generator, mask_generator_right_lung, mask_generator_left_lung, mask_generator_heart,
                    mask_generator_left_clavicle, mask_generator_right_clavicle,
                    uncertainty_generator, mask_generator_right_lung_pred, mask_generator_left_lung_pred,
                    mask_generator_heart_pred,
                    mask_generator_left_clavicle_pred, mask_generator_right_clavicle_pred)

    for (img, mask_right_lung, mask_left_lung, mask_heart, mask_left_clavicle, mask_right_clavicle,
         uncertainty, mask_right_lung_pred, mask_left_lung_pred, mask_heart_pred, mask_left_clavicle_pred,
         mask_right_clavicle_pred) in train_gen:
        img, mask_right_lung, mask_left_lung, mask_heart, mask_left_clavicle, mask_right_clavicle = resize_data_multi(
            img, mask_right_lung, mask_left_lung, mask_heart, mask_left_clavicle, mask_right_clavicle, 224)
        uncertainty, mask_right_lung_pred, mask_left_lung_pred, mask_heart_pred, mask_left_clavicle_pred, mask_right_clavicle_pred = resize_data_multi(
            uncertainty, mask_right_lung_pred, mask_left_lung_pred, mask_heart_pred, mask_left_clavicle_pred,
            mask_right_clavicle_pred, 224)
        # print('adjust img shape ', img.shape)
        img = img[:, :, :, 0]
        uncertainty = uncertainty[:, :, :, 0]

        mask_right_lung = mask_right_lung[:, :, :, 0]
        mask_left_lung = mask_left_lung[:, :, :, 0]
        mask_heart = mask_heart[:, :, :, 0]
        mask_left_clavicle = mask_left_clavicle[:, :, :, 0]
        mask_right_clavicle = mask_right_clavicle[:, :, :, 0]

        mask_right_lung_pred = mask_right_lung_pred[:, :, :, 0]
        mask_left_lung_pred = mask_left_lung_pred[:, :, :, 0]
        mask_heart_pred = mask_heart_pred[:, :, :, 0]
        mask_left_clavicle_pred = mask_left_clavicle_pred[:, :, :, 0]
        mask_right_clavicle_pred = mask_right_clavicle_pred[:, :, :, 0]

        mask = np.zeros(mask_heart.shape + (7,))  # 5 class (except background), 1 img, 1 uncertainty
        quality_value = np.zeros((1, 1))
        # print('img shape ', img.shape)
        # print('uncertainty shape ', uncertainty.shape)
        # print('mask shape 0 ', mask[:, :, :, 0].shape)
        # print('mask shape ', mask.shape)
        mask[:, :, :, 0] = img
        mask[:, :, :, 1] = uncertainty
        mask[mask_right_lung_pred == 1, 2] = 1
        quality_value = np.add(quality_value, jaccard_index(mask_right_lung, mask_right_lung_pred))
        mask[mask_left_lung_pred == 1, 3] = 1
        quality_value = np.add(quality_value, jaccard_index(mask_left_lung, mask_left_lung_pred))
        mask[mask_heart_pred == 1, 4] = 1
        quality_value = np.add(quality_value, jaccard_index(mask_heart, mask_heart_pred))
        mask[mask_left_clavicle_pred == 1, 5] = 1
        quality_value = np.add(quality_value, jaccard_index(mask_left_clavicle, mask_left_clavicle_pred))
        mask[mask_right_clavicle_pred == 1, 6] = 1
        quality_value = np.add(quality_value, jaccard_index(mask_right_clavicle, mask_right_clavicle_pred))

        yield (mask, np.divide(quality_value, 5))

def adjust_img(mask):

    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    return mask

def resize_data_multi(img,mask1, mask2, mask3, mask4, mask5, size):
    shape = (size, size)
    img_ = np.zeros((1, size, size,1))
    img_[0,:,:,0] = cv2.resize(img[0,:,:], shape)
    img_ = img_ / 255

    mask1_ = np.zeros((1, size, size,1))
    mask1_[0,:,:,0] = cv2.resize(mask1[0,:,:], shape)
    mask1_ = mask1_ / 255
    mask1_[mask1_ > 0.5] = 1
    mask1_[mask1_ <= 0.5] = 0

    mask2_ = np.zeros((1, size, size,1))
    mask2_[0,:,:,0] = cv2.resize(mask2[0,:,:], shape)
    mask2_ = mask2_ / 255
    mask2_[mask2_ > 0.5] = 1
    mask2_[mask2_ <= 0.5] = 0

    mask3_ = np.zeros((1, size, size,1))
    mask3_[0,:,:,0] = cv2.resize(mask3[0,:,:], shape)
    mask3_ = mask3_ / 255
    mask3_[mask3_ > 0.5] = 1
    mask3_[mask3_ <= 0.5] = 0

    mask4_ = np.zeros((1, size, size,1))
    mask4_[0,:,:,0] = cv2.resize(mask4[0,:,:], shape)
    mask4_ = mask4_ / 255
    mask4_[mask4_ > 0.5] = 1
    mask4_[mask4_ <= 0.5] = 0

    mask5_ = np.zeros((1, size, size,1))
    mask5_[0,:,:,0] = cv2.resize(mask5[0,:,:], shape)
    mask5_ = mask5_ / 255
    mask5_[mask5_ > 0.5] = 1
    mask5_[mask5_ <= 0.5] = 0


    return (img_, mask1_, mask2_, mask3_, mask4_, mask5_)

