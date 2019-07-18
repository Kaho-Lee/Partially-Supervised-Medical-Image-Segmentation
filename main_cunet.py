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

from mnet_utils import mk_dir, return_list, train_loader
import UNet_CUNet
from Utils import *

import numpy as np
import shutil
import os
import cv2
import matplotlib.pyplot as plt

def add_suffix(base_file, suffix):
    filename, fileext = os.path.splitext(base_file)
    return "%s_%s%s" % (filename, suffix, fileext)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)

BATCH_SIZE=1

EPOCHS=80
EPISODES = 15
#result_path = mk_dir('./deep_model/')
# pre_model_file = './deep_model/Model_MNet_REFUGE.h5'


root_path = '../UNetJSRT/'
data_path = root_path + 'data/'
mask_path_heart = root_path + 'dilate/heart/'
mask_path_right_lung = root_path + 'dilate/right_lung/'
mask_path_left_lung = root_path + 'dilate/left_lung/'
mask_path_left_clavicle = root_path + 'dilate/left_clavicle/'
mask_path_right_clavicle = root_path + 'dilate/right_clavicle/'
data_type = '.png'
# save_model_file = root_path + 'Model_UNet_JSRT.h5'

save_model_file =  root_path + 'Model/'+ 'UNet_Partial_FullModel_CUNet_Adam.h5'

loss_train = np.zeros((EPOCHS, EPISODES*2))
acc_train = np.zeros((EPOCHS, EPISODES*2))
dice_train = np.zeros((EPOCHS,EPISODES*2))

for j in range(2):
    np.random.seed(j)

    for episode in range(EPISODES):
        print(str(episode+1)+"/"+str(EPISODES))
        Index = list(range(0,247))
        valIndex = list(np.random.randint(0,246,27))
        trainIndex = [x for x in Index if x not in valIndex]

        # val_data_path = root_path + 'val_data/data/'
        # val_mask_path = root_path + 'val_data/label/'
        file_list = return_list(data_path, data_type)
        train_list = [file_list[x] for x in trainIndex]
        val_list = [file_list[x] for x in valIndex]
        print('training number', len(train_list))
        print('val number', len(val_list))
        SegAug = os.path.join(root_path, 'SegAug')


        if(os.path.exists(os.path.join(root_path, 'train1'))):
            shutil.rmtree(os.path.join(root_path, 'train1'))
        if(os.path.exists(os.path.join(root_path, 'val1'))):
            shutil.rmtree(os.path.join(root_path, 'val1'))

        if(os.path.exists(os.path.join(root_path, 'SegTrain1'))):
            shutil.rmtree(os.path.join(root_path, 'SegTrain1'))
        if(os.path.exists(os.path.join(root_path, 'SegAug1'))):
            shutil.rmtree(os.path.join(root_path, 'SegAug1'))

        mk_dir(root_path+'train1/')
        mk_dir(root_path+'train1/'+'IMG/')
        mk_dir(root_path+'train1/'+'right_lung/')
        mk_dir(root_path+'train1/'+'left_lung/')
        mk_dir(root_path+'train1/'+'heart/')
        mk_dir(root_path+'train1/'+'left_clavicle/')
        mk_dir(root_path+'train1/'+'right_clavicle/')



        mk_dir(root_path+'val1/')
        mk_dir(root_path+'val1/'+'IMG/')
        mk_dir(root_path+'val1/'+'right_lung/')
        mk_dir(root_path+'val1/'+'left_lung/')
        mk_dir(root_path+'val1/'+'heart/')
        mk_dir(root_path+'val1/'+'left_clavicle/')
        mk_dir(root_path+'val1/'+'right_clavicle/')


        mk_dir(root_path+'SegAug1/')

        for item in train_list:
            image = cv2.imread(data_path+item)
            cv2.imwrite(root_path+'train1/'+'IMG/'+item, image)

            image = cv2.imread(mask_path_right_lung+item)
            cv2.imwrite(root_path+'train1/'+'right_lung/'+item, image)

            image = cv2.imread(mask_path_left_lung+item)
            cv2.imwrite(root_path+'train1/'+'left_lung/'+item, image)

            image = cv2.imread(mask_path_heart+item)
            cv2.imwrite(root_path+'train1/'+'heart/'+item, image)

            image = cv2.imread(mask_path_left_clavicle+item)
            cv2.imwrite(root_path+'train1/'+'left_clavicle/'+item, image)

            image = cv2.imread(mask_path_right_clavicle+item)
            cv2.imwrite(root_path+'train1/'+'right_clavicle/'+item, image)

        for item in val_list:
            image = cv2.imread(data_path+item)
            cv2.imwrite(root_path+'val1/'+'IMG/'+item, image)

            image = cv2.imread(mask_path_right_lung+item)
            cv2.imwrite(root_path+'val1/'+'right_lung/'+item, image)

            image = cv2.imread(mask_path_left_lung+item)
            cv2.imwrite(root_path+'val1/'+'left_lung/'+item, image)

            image = cv2.imread(mask_path_heart+item)
            cv2.imwrite(root_path+'val1/'+'heart/'+item, image)

            image = cv2.imread(mask_path_left_clavicle+item)
            cv2.imwrite(root_path+'val1/'+'left_clavicle/'+item, image)

            image = cv2.imread(mask_path_right_clavicle+item)
            cv2.imwrite(root_path+'val1/'+'right_clavicle/'+item, image)

        # val_img = cv2.imread(root_path+'val/'+'IMG/'+return_list(root_path+'val/IMG/', '.png')[0])
        # val_mask_lung = cv2.imread(root_path+'val/'+'right_lung/'+return_list(root_path+'val/'+'right_lung/', '.png')[0])
        # val_mask_heart = cv2.imread(root_path+'val/'+'heart/'+return_list(root_path+'val/'+'heart/', '.png')[0])
        # val_img, val_mask_lung, val_mask_heart = adjust_data_multi(val_img, val_mask_lung, val_mask_heart)
        # back_lung = np.zeros(val_mask_lung.shape)
        # back_lung[val_mask_lung==0] = 1
        # back_heart = np.zeros(val_mask_heart.shape)
        # back_heart[val_mask_heart==0] = 1
        # back = np.multiply(back_lung, back_heart)
        # fig, axs = plt.subplots(1, 3, figsize=(15, 8))
        # axs[0].imshow(val_mask_lung)
        # axs[1].imshow(val_mask_heart)
        # axs[2].imshow(back)
        # plt.show()
        # a=b



        input_size = 512

        # print(train_list, len(train_list))
        # print(val_list, len(val_list))

        optimizer_SGD = SGD(lr=0.0001, momentum=0.9)
        Optimizer_Adam = Adam(lr=5e-5)

        UNetModel = UNet_CUNet.DeepModel(size_set=input_size)
        UNetModel.compile(optimizer=Optimizer_Adam, loss=dice_coef_loss_partial_annotated, metrics=[dice_coef, 'accuracy'])

        # UNetModel.compile(optimizer=Optimizer_Adam, loss=dice_coef_loss_partial_annotated, metrics=[dice_coef, 'accuracy'])
        # UNetModel.compile(optimizer=Optimizer_Adam, loss=binary_crossentropy_jiahao, metrics=[dice_coef, 'accuracy'])
        UNetModel.summary()

        train_generator_args = dict(rotation_range=0.2,
                                    width_shift_range=0.05,
                                    height_shift_range=0.05,
                                    shear_range=0.05,
                                    zoom_range=0.05,
                                    horizontal_flip=True,
                                    fill_mode='nearest')

        train_gen = train_generator_multiStruct(BATCH_SIZE,
                                    root_path+'train1',
                                    'IMG',
                                    'right_lung',
                                    'left_lung',
                                    'heart',
                                    'left_clavicle',
                                    'right_clavicle',
                                    train_generator_args,
                                    target_size=(512,512))

        validation_data = (test_load_image(data_path+val_list[0], target_size=(512, 512)),
                            test_load_image(mask_path_right_lung+val_list[0], target_size=(512, 512)))

        val_gen = val_generator_multiStruct(BATCH_SIZE,
                                    root_path+'val1',
                                    'IMG',
                                    'right_lung',
                                    'left_lung',
                                    'heart',
                                    'left_clavicle',
                                    'right_clavicle',
                                    train_generator_args,
                                    target_size=(512,512))

        # model_checkpoint = ModelCheckpoint('unet_lung_seg.hdf5',
        #                                    monitor='loss',
        #                                    verbose=1,
        #                                    save_best_only=True)

        history = UNetModel.fit_generator(train_gen,
                                      steps_per_epoch=len(train_list) / BATCH_SIZE,
                                      epochs=EPOCHS,
                                      validation_data = val_gen,
                                      validation_steps=len(val_list))

        # history = UNetModel.fit_generator(
        #     generator=train_loader(train_list, root_path+'train/'+'IMG/', root_path+'train/'+'mask/', input_size),
        #     steps_per_epoch=len(train_list),
        #     validation_data=train_loader(val_list, root_path+'val/'+'IMG/', root_path+'val/'+'mask/', input_size),
        #     validation_steps=len(train_list),
        #     verbose=0,
        #     epochs=EPOCHS
        # )
        UNetModel.save(save_model_file)
        fig, axs = plt.subplots(1, 2, figsize = (15, 4))

        UNetModel.save(save_model_file)


        loss_train[:,episode+j*EPISODES] = history.history['loss']
        validation_loss = history.history['val_loss']

        acc_train[:,episode+j*EPISODES] = history.history['acc']
        validation_accuracy = history.history['val_acc']

        dice_train[:,episode+j*EPISODES] = history.history['dice_coef']
        validation_dice = history.history['val_dice_coef']

        # epoch_count = range(1, len(training_loss) + 1)

        EXp_dice = open('EXP_dice_CUNet.txt', "a")
        EXp_dice.writelines([str(validation_dice[len(validation_dice)-1]), "\n"])
        EXp_dice.close()

        EXp_acc = open('EXP_acc_CUNet.txt', "a")
        EXp_acc.writelines([str(validation_accuracy[len(validation_accuracy)-1]), "\n"])
        EXp_acc.close()


loss_train = np.mean(loss_train, axis=0)
acc_train  = np.mean(acc_train, axis=0)
dice_train = np.mean(dice_train, axis=0)

LossFile = open('Loss_CUNet.txt', "w")
for i in range(EPOCHS):
   LossFile.writelines([str(loss_train[i]), "\n"])
LossFile.close()

AccFile = open('Accuracy_CUNet.txt', "w")
for i in range(EPOCHS):
   AccFile.writelines([str(acc_train[i]), " ",str(dice_train[i]), "\n"])
AccFile.close()

fig, axs = plt.subplots(1, 2, figsize = (15, 4))
axs[0].plot(EPOCHS, loss_train, 'r--')
axs[0].legend(['Training Loss'])


axs[1].plot(EPOCHS, acc_train, 'r--')
axs[1].plot(EPOCHS, dice_train, 'b-')
axs[1].legend(['Training Accuracy', 'Training Dice Score'])
plt.savefig("Traing_Error_Accuracy_partial_CUNet")
