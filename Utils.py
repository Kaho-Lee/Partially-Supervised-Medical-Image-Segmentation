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
from keras import utils as np_utils
import keras.backend as K
from glob import glob
from tqdm import tqdm

def add_colored_dilate(image, mask_image, dilate_image):
    mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    dilate_image_gray = cv2.cvtColor(dilate_image, cv2.COLOR_BGR2GRAY)

    mask = cv2.bitwise_and(mask_image, mask_image, mask=mask_image_gray)
    dilate = cv2.bitwise_and(dilate_image, dilate_image, mask=dilate_image_gray)

    mask_coord = np.where(mask!=[0,0,0])
    dilate_coord = np.where(dilate!=[0,0,0])

    mask[mask_coord[0],mask_coord[1],:]=[255,0,0]
    dilate[dilate_coord[0],dilate_coord[1],:] = [0,0,255]

    ret = cv2.addWeighted(image, 0.7, dilate, 0.3, 0)
    ret = cv2.addWeighted(ret, 0.7, mask, 0.3, 0)

    return ret

def add_colored_mask(image, mask_image):
    mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)

    mask = cv2.bitwise_and(mask_image, mask_image, mask=mask_image_gray)

    mask_coord = np.where(mask!=[0,0,0])

    mask[mask_coord[0],mask_coord[1],:]=[255,0,0]

    ret = cv2.addWeighted(image, 0.7, mask, 0.3, 0)

    return ret

def add_colored_mask_JSRT(image, mask_left_lung, mask_right_lung, mask_heart, mask_left_clavicle, mask_right_clavicle):
    mask_left_lung_gray = cv2.cvtColor(mask_left_lung, cv2.COLOR_BGR2GRAY)
    mask_right_lung_gray = cv2.cvtColor(mask_right_lung, cv2.COLOR_BGR2GRAY)
    mask_heart_gray = cv2.cvtColor(mask_heart, cv2.COLOR_BGR2GRAY)
    mask_left_clavicle_gray = cv2.cvtColor(mask_left_clavicle, cv2.COLOR_BGR2GRAY)
    mask_right_clavicle_gray = cv2.cvtColor(mask_right_clavicle, cv2.COLOR_BGR2GRAY)

    mask_left_lung = cv2.bitwise_and(mask_left_lung, mask_left_lung, mask=mask_left_lung_gray)
    mask_right_lung = cv2.bitwise_and(mask_right_lung, mask_right_lung, mask=mask_right_lung_gray)
    mask_heart = cv2.bitwise_and(mask_heart, mask_heart, mask=mask_heart_gray)
    mask_left_clavicle = cv2.bitwise_and(mask_left_clavicle, mask_left_clavicle, mask=mask_left_clavicle_gray)
    mask_right_clavicle = cv2.bitwise_and(mask_right_clavicle, mask_right_clavicle, mask=mask_right_clavicle_gray)

    mask_coord_left_lung = np.where(mask_left_lung!=[0,0,0])
    mask_coord_right_lung = np.where(mask_right_lung!=[0,0,0])
    mask_coord_heart = np.where(mask_heart!=[0,0,0])
    mask_coord_left_clavicle = np.where(mask_left_clavicle!=[0,0,0])
    mask_coord_right_clavicle = np.where(mask_right_clavicle!=[0,0,0])

    mask_left_lung[mask_coord_left_lung[0],mask_coord_left_lung[1],:]=[255,0,0]
    mask_right_lung[mask_coord_right_lung[0],mask_coord_right_lung[1],:]=[255,0,0]
    mask_heart[mask_coord_heart[0],mask_coord_heart[1],:]=[0,0,255]

    mask_left_clavicle[mask_coord_left_clavicle[0],mask_coord_left_clavicle[1],:]=[0,255,0]
    mask_right_clavicle[mask_coord_right_clavicle[0],mask_coord_right_clavicle[1],:]=[0,255,0]

    mask = cv2.add(mask_left_lung, mask_right_lung)
    mask = cv2.add(mask, mask_heart)
    mask = cv2.add(mask, mask_left_clavicle)
    mask = cv2.add(mask, mask_right_clavicle)

    ret = cv2.addWeighted(image, 0.7, mask, 0.3, 0)

    return ret

def diff_mask(ref_image, mask_image):
    mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)

    mask = cv2.bitwise_and(mask_image, mask_image, mask=mask_image_gray)

    mask_coord = np.where(mask!=[0,0,0])

    mask[mask_coord[0],mask_coord[1],:]=[255,0,0]

    ret = cv2.addWeighted(ref_image, 0.7, mask, 0.3, 0)
    return ret


'''
    Taken from https://github.com/zhixuhao/unet/blob/master/data.py
'''
def test_load_image(test_file, target_size=(256,256)):
    img = cv2.imread(test_file, cv2.IMREAD_GRAYSCALE)
    img = img / 255
    img = cv2.resize(img, target_size)
    img = np.reshape(img, img.shape + (1,))
    img = np.reshape(img,(1,) + img.shape)
    return img

def test_generator(test_path, test_files, target_size=(256,256)):
    for test_file in test_files:
        yield test_load_image(test_path+test_file, target_size)

def save_result(save_path, npyfile, test_files):
    for i, item in enumerate(npyfile):
        result_file = test_files[i]
        img = (item[:, :, 0] * 255.).astype(np.uint8)

        filename, fileext = os.path.splitext(os.path.basename(result_file))

        result_file = os.path.join(save_path, "%s%s" % (filename, fileext))

        cv2.imwrite(result_file, img)

def mk_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path



def save_result_multilabel(save_path, npyfile, test_files, numClass):
    for i in range(numClass):
        classFolder = 'class'+str(i+1)+'/'
        if(os.path.exists(os.path.join(save_path, classFolder))):
            shutil.rmtree(os.path.join(save_path, classFolder))
        mk_dir(save_path + classFolder)


    for i, item in enumerate(npyfile):
        result_file = test_files[i]
        filename, fileext = os.path.splitext(os.path.basename(result_file))

        for j in range(numClass):
            classFolder = 'class'+str(j+1)+'/'
            # arg_max = np.argmax(item, axis = 2)
            img = np.zeros((512,512,1))
            # img[arg_max==i, 0] = 1
            img[:,:,0] = item[:,:,j]
            img[img>0.5] = 1
            img[img<=0.5] = 0
            img = (img[:,:,0] * 255.).astype(np.uint8)
            save_path_img = save_path + classFolder
            result_file = os.path.join(save_path_img, "%s%s" % (filename, fileext))
            cv2.imwrite(result_file, img)

def save_result_multiclass(save_path, npyfile, test_files, numClass):
    for i in range(numClass):
        classFolder = 'class'+str(i+1)+'/'
        if(os.path.exists(os.path.join(save_path, classFolder))):
            shutil.rmtree(os.path.join(save_path, classFolder))
        mk_dir(save_path + classFolder)


    for i, item in enumerate(npyfile):
        result_file = test_files[i]
        filename, fileext = os.path.splitext(os.path.basename(result_file))

        for i in range(numClass):
            classFolder = 'class'+str(i+1)+'/'
            arg_max = np.argmax(item, axis = 2)
            img = np.zeros((512,512,1))
            img[arg_max==i, 0] = 1
            img = (img[:, :, 0] * 255.).astype(np.uint8)
            save_path_img = save_path + classFolder
            result_file = os.path.join(save_path_img, "%s%s" % (filename, fileext))
            cv2.imwrite(result_file, img)

def partial_binary_crossentropy(y_true, y_pred):
    partial_pred = np.zeros((512, 512, 3))
    partial_true = np.zeros((512, 512, 3))

    # partial_pred[:,:,0] = y_pred[:,:,0]
    # partial_pred[:,:,1] = y_pred[:,:,1]
    # partial_pred[:,:,2] = y_pred[:,:,5]
    # partial_true[:,:,0] = y_true[:,:,0]
    # partial_true[:,:,1] = y_true[:,:,1]
    # partial_true[:,:,2] = y_true[:,:,5]

    partial = []
    partial.append(keras.binary_crossentropy(y_true[:,:,0], y_pred[:,:,0]))
    partial.append(keras.binary_crossentropy(y_true[:,:,1], y_pred[:,:,1]))
    partial.append(keras.binary_crossentropy(y_true[:,:,5], y_pred[:,:,5]))
    result_partial = np.mean(partial)

    # result_partial = keras.binary_crossentropy(partial_true, partial_pred)
    result_full = keras.binary_crossentropy(y_true, y_pred)

    a = keras.sum(y_true[:,:,:,2])
    b = keras.equal(a, 0)

    return keras.switch(b, result_partial, result_full)


def dice_coef(y_true, y_pred):
    # smooth = 1.
    # y_true_f = K.flatten(y_true)
    # y_pred_f = K.flatten(y_pred)
    # intersection = K.sum(y_true_f * y_pred_f)
    # return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

def dice_coef_loss_partial_annotated(y_true, y_pred):
    numclass = 6
    missing_mask = keras.variable(np.zeros((1,512,512,1)))
    print("new loss func")
    print("y_true shape", y_true.shape)
    #a = t.keras.backend.eval(y_true)
    a = keras.sum(y_true[:,:,:,2])
    b = keras.equal(a, 0)
        #print('loss for mannually eliminate annotation')
        #LossFile = open('new_loss.txt', "w")
        #LossFile.writelines(["In new loss func with missing annotation"])
        #LossFile.close()

    #b  = keras.print_tensor(b, message="\nx is: ")

    '''
    partial annotated
    '''
    score0 = dice_coef(y_true[:, :, :, 0], y_pred[:, :, :, 0])
    score1 = dice_coef(y_true[:, :, :, 1], y_pred[:, :, :, 1])
    score5 = dice_coef(y_true[:, :, :, 5], y_pred[:, :, :, 5])
    loss1 = -(score0+score1+score5)/(numclass -3)
        #print("kk")
        #tf.enable_eager_execution()
        #tf.print("Inside new loss function with missing anno")
        #y_true = keras.print_tensor(y_true, message="x is: ")
    '''
    fully supervised
    '''
    #score0 = dice_coef(y_true[:, :, :, 0], y_pred[:, :, :, 0])
    #score1 = dice_coef(y_true[:, :, :, 1], y_pred[:, :, :, 1])
    score2 = dice_coef(y_true[:, :, :, 2], y_pred[:, :, :, 2])
    score3 = dice_coef(y_true[:, :, :, 3], y_pred[:, :, :, 3])
    score4 = dice_coef(y_true[:, :, :, 4], y_pred[:, :, :, 4])
    #score5 = dice_coef(y_true[:, :, :, 5], y_pred[:, :, :, 5])
    loss2 = -(score0+score1+score2+score3+score4+score5)/numclass
    return keras.switch(b, loss1, loss2)

def dice_coef2(y_true, y_pred):
    score0 = dice_coef(y_true[:, :, :, 0], y_pred[:, :, :, 0])
    score1 = dice_coef(y_true[:, :, :, 1], y_pred[:, :, :, 1])
    score = 0.5 * score0 + 0.5 * score1

    return score


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

    # return -dice_coef2(y_true, y_pred)

"""
    Taken from From: https://github.com/zhixuhao/unet/blob/master/data.py on 0309, 2019
"""
def train_generator(batch_size, train_path, image_folder, mask_folder, aug_dict,
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
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)

    print('111111')

    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)

    train_gen = zip(image_generator, mask_generator)

    for (img, mask) in train_gen:
        img, mask = adjust_data(img, mask)
        # print('mask shape ', mask.shape)
        yield (img,mask)

def val_generator(batch_size, train_path, image_folder, mask_folder, aug_dict,
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
    mask_datagen = ImageDataGenerator()

    print('111111')

    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)

    train_gen = zip(image_generator, mask_generator)

    for (img, mask) in train_gen:
        img, mask = adjust_data(img, mask)
        # print('mask shape ', mask.shape)
        yield (img,mask)

def train_generator_multiStruct(batch_size, train_path, image_folder,
        mask_right_lung_folder, mask_left_lung_folder, mask_heart_folder,
        mask_left_clavicle_folder, mask_right_clavicle_folder,
        aug_dict,
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
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen_right_lung = ImageDataGenerator(**aug_dict)
    mask_datagen_left_lung = ImageDataGenerator(**aug_dict)
    mask_datagen_heart = ImageDataGenerator(**aug_dict)
    mask_datagen_left_clavicle  = ImageDataGenerator(**aug_dict)
    mask_datagen_right_clavicle = ImageDataGenerator(**aug_dict)

    print('111111')

    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)


    mask_generator_right_lung = mask_datagen_right_lung.flow_from_directory(
        train_path,
        classes = [mask_right_lung_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)

    mask_generator_left_lung = mask_datagen_right_lung.flow_from_directory(
        train_path,
        classes = [mask_left_lung_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)

    mask_generator_heart = mask_datagen_heart.flow_from_directory(
        train_path,
        classes = [mask_heart_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)

    mask_generator_left_clavicle = mask_datagen_heart.flow_from_directory(
        train_path,
        classes = [mask_left_clavicle_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)

    mask_generator_right_clavicle = mask_datagen_heart.flow_from_directory(
        train_path,
        classes = [mask_right_clavicle_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)

    train_gen = zip(image_generator, mask_generator_right_lung, mask_generator_left_lung, mask_generator_heart,
                        mask_generator_left_clavicle, mask_generator_right_clavicle)

    for (img, mask_right_lung, mask_left_lung, mask_heart, mask_left_clavicle, mask_right_clavicle) in train_gen:
        img, mask_right_lung, mask_left_lung, mask_heart, mask_left_clavicle, mask_right_clavicle = adjust_data_multi(img, mask_right_lung, mask_left_lung, mask_heart, mask_left_clavicle, mask_right_clavicle)
        # mask_combine = np.vstack((mask_right_lung, mask_heart))
        # print('ori shape ', mask_right_lung.shape)

        # mask = np.zeros((1,target_size[0], target_size[1], 2), dtype=np.bool)
        mask_right_lung = mask_right_lung[:,:,:,0]
        mask_left_lung = mask_left_lung[:,:,:,0]
        mask_heart = mask_heart[:,:,:,0]
        mask_left_clavicle = mask_left_clavicle[:,:,:,0]
        mask_right_clavicle = mask_right_clavicle[:,:,:,0]

        mask = np.zeros(mask_heart.shape + (6,)) #6 = num of class
        mask[mask_right_lung==1, 0] = 1
        mask[mask_left_lung==1, 1] = 1

        flip_coin = np.random.uniform(0,1)
        if(flip_coin >= 0.2):
            #print('\nhava heart anno')
            mask[mask_heart==1, 2] = 1
            mask[mask_left_clavicle==1, 3] = 1
            mask[mask_right_clavicle==1, 4] = 1


            back_right_lung = np.zeros(mask_heart.shape)
            back_right_lung[mask_right_lung==0] = 1

            back_left_lung = np.zeros(mask_heart.shape)
            back_left_lung[mask_left_lung==0] = 1

            back_heart = np.zeros(mask_heart.shape)
            back_heart[mask_heart==0] = 1

            back_left_clavicle = np.zeros(mask_heart.shape)
            back_left_clavicle[mask_left_clavicle==0] = 1

            back_right_clavicle = np.zeros(mask_heart.shape)
            back_right_clavicle[mask_right_clavicle==0] = 1


            back = np.multiply(back_right_lung, back_heart)
            back = np.multiply(back, back_left_lung)
            back = np.multiply(back, back_left_clavicle)
            back = np.multiply(back, back_right_clavicle)
            mask[:,:,:, 5] = back
           # yield(img, mask)
        else:
            #print('\nmanually eliminate heart annotation')
           # missing_mask = np.zeros((512,512,1))
            #print('loss for mannually eliminate annotation ',np.array_equal(mask[0,:,:,1], missing_mask[:,:,0]))
            #print('missing anno ', mask[:,:,:,1].shape,mask[:,:,:,1])
            #print('missing mask ', missing_mask[:,:,0].shape,missing_mask[:,:,0])
            back_right_lung = np.zeros(mask_heart.shape)
            back_right_lung[mask_right_lung==0] = 1
            back_left_lung = np.zeros(mask_heart.shape)
            back_left_lung[mask_left_lung==0] = 1
            mask[:,:,:,5] =np.multiply(back_right_lung, back_left_lung)

            # mask[:,:,:, 2] = back_lung
        yield(img, mask)



        # mask = np_utils.to_categorical(mask, 3)
        # mask = mask.reshape((1,512,512,3))
        # mask = np_utils.to_categorical(mask, 2)
        # print('mask shape ', mask.shape)
        # yield (img, [mask_right_lung, mask_heart])


def val_generator_multiStruct(batch_size, train_path, image_folder,
        mask_right_lung_folder, mask_left_lung_folder, mask_heart_folder,
        mask_left_clavicle_folder, mask_right_clavicle_folder,
        aug_dict,
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


    print('111111')

    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)

    mask_generator_right_lung = mask_datagen_right_lung.flow_from_directory(
        train_path,
        classes = [mask_right_lung_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)

    mask_generator_left_lung = mask_datagen_right_lung.flow_from_directory(
        train_path,
        classes = [mask_left_lung_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)

    mask_generator_heart = mask_datagen_heart.flow_from_directory(
        train_path,
        classes = [mask_heart_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)

    mask_generator_left_clavicle = mask_datagen_heart.flow_from_directory(
        train_path,
        classes = [mask_left_clavicle_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)

    mask_generator_right_clavicle = mask_datagen_heart.flow_from_directory(
        train_path,
        classes = [mask_right_clavicle_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)

    train_gen = zip(image_generator, mask_generator_right_lung, mask_generator_left_lung, mask_generator_heart,
                        mask_generator_left_clavicle, mask_generator_right_clavicle)

    for (img, mask_right_lung, mask_left_lung, mask_heart, mask_left_clavicle, mask_right_clavicle) in train_gen:
        img, mask_right_lung, mask_left_lung, mask_heart, mask_left_clavicle, mask_right_clavicle = adjust_data_multi(img, mask_right_lung, mask_left_lung, mask_heart, mask_left_clavicle, mask_right_clavicle)
        # mask_combine = np.vstack((mask_right_lung, mask_heart))
        # print('ori shape ', mask_right_lung.shape)

        # mask = np.zeros((1,target_size[0], target_size[1], 2), dtype=np.bool)
        mask_right_lung = mask_right_lung[:,:,:,0]
        mask_left_lung = mask_left_lung[:,:,:,0]
        mask_heart = mask_heart[:,:,:,0]
        mask_left_clavicle = mask_left_clavicle[:,:,:,0]
        mask_right_clavicle = mask_right_clavicle[:,:,:,0]

        mask = np.zeros(mask_heart.shape + (6,)) #6 = num of class
        mask[mask_right_lung==1, 0] = 1
        mask[mask_left_lung==1, 1] = 1
        mask[mask_heart==1, 2] = 1
        mask[mask_left_clavicle==1, 3] = 1
        mask[mask_right_clavicle==1, 4] =1


        back_right_lung = np.zeros(mask_heart.shape)
        back_right_lung[mask_right_lung==0] = 1

        back_left_lung = np.zeros(mask_heart.shape)
        back_left_lung[mask_left_lung==0] = 1

        back_heart = np.zeros(mask_heart.shape)
        back_heart[mask_heart==0] = 1

        back_left_clavicle = np.zeros(mask_heart.shape)
        back_left_clavicle[mask_left_clavicle==0] = 1

        back_right_clavicle = np.zeros(mask_heart.shape)
        back_right_clavicle[mask_right_clavicle==0] = 1


        back = np.multiply(back_right_lung, back_heart)
        back = np.multiply(back, back_left_lung)
        back = np.multiply(back, back_left_clavicle)
        back = np.multiply(back, back_right_clavicle)
        mask[:,:,:, 5] = back


        # mask = np_utils.to_categorical(mask, 3)
        # mask = mask.reshape((1,512,512,3))
        # cv2.imwrite("UNetJSRT/multiMask/1.png", back_lung)
        # cv2.imwrite("UNetJSRT/multiMask/2.png", back_heart)
        # cv2.imwrite("UNetJSRT/multiMask/3.png", back)

        # mask = np_utils.to_categorical(mask, 2)
        # print('mask shape ', mask.shape)
        # yield (img, [mask_right_lung, mask_heart])
        yield(img, mask)

def adjust_data(img,mask):
    img = img / 255
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    return (img, mask)


def adjust_data_multi(img,mask1, mask2, mask3, mask4, mask5):
    img = img / 255

    mask1 = mask1 / 255
    mask1[mask1 > 0.5] = 1
    mask1[mask1 <= 0.5] = 0

    mask2 = mask2 / 255
    mask2[mask2 > 0.5] = 1
    mask2[mask2 <= 0.5] = 0

    mask3 = mask3 / 255
    mask3[mask3 > 0.5] = 1
    mask3[mask3 <= 0.5] = 0

    mask4 = mask4 / 255
    mask4[mask4 > 0.5] = 1
    mask4[mask4 <= 0.5] = 0

    mask5 = mask5 / 255
    mask5[mask5 > 0.5] = 1
    mask5[mask5 <= 0.5] = 0


    return (img, mask1, mask2, mask3, mask4, mask5)
