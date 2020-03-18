'''
Sample code for the main file of model training
'''

import random
from keras.optimizers import SGD
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

import UNet
from Utils import train_generator, test_load_image, dice_coef_loss, dice_coef, train_generator, val_generator

import numpy as np
import shutil
import os
import cv2
import matplotlib.pyplot as plt

def add_suffix(base_file, suffix):
    filename, fileext = os.path.splitext(base_file)
    return "%s_%s%s" % (filename, suffix, fileext)

BATCH_SIZE=2

EPOCHS=2

# result_path = mk_dir('./deep_model/')
# pre_model_file = './deep_model/Model_MNet_REFUGE.h5'


root_path = '../UNetJSRT/'
data_path = root_path + 'data/'
mask_path = root_path + 'dilate/heart/'
data_type = '.png'
save_model_file = root_path + 'Model_UNet_JSRT.h5'

save_model_file =  root_path + 'Model/'+ 'UNetHeart_Adam.h5'

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


if(os.path.exists(os.path.join(root_path, 'train'))):
    shutil.rmtree(os.path.join(root_path, 'train'))
if(os.path.exists(os.path.join(root_path, 'val'))):
    shutil.rmtree(os.path.join(root_path, 'val'))

if(os.path.exists(os.path.join(root_path, 'SegTrain'))):
    shutil.rmtree(os.path.join(root_path, 'SegTrain'))
if(os.path.exists(os.path.join(root_path, 'SegAug'))):
    shutil.rmtree(os.path.join(root_path, 'SegAug'))

mk_dir(root_path+'train/')
mk_dir(root_path+'train/'+'IMG/')
mk_dir(root_path+'train/'+'mask/')

mk_dir(root_path+'val/')
mk_dir(root_path+'val/'+'IMG/')
mk_dir(root_path+'val/'+'mask/')

mk_dir(root_path+'SegAug/')

for item in train_list:
    image = cv2.imread(data_path+item)
    cv2.imwrite(root_path+'train/'+'IMG/'+item, image)
    image = cv2.imread(mask_path+item)
    cv2.imwrite(root_path+'train/'+'mask/'+item, image)

for item in val_list:
    image = cv2.imread(data_path+item)
    cv2.imwrite(root_path+'val/'+'IMG/'+item, image)
    image = cv2.imread(mask_path+item)
    cv2.imwrite(root_path+'val/'+'mask/'+item, image)



input_size = 512

# print(train_list, len(train_list))
# print(val_list, len(val_list))

optimizer_SGD = SGD(lr=0.0001, momentum=0.9)
Optimizer_Adam = Adam(lr=1e-5)

UNetModel = UNet.DeepModel(size_set=input_size)
UNetModel.compile(optimizer=Optimizer_Adam, loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy'])
UNetModel.summary()

train_generator_args = dict(rotation_range=0.2,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            shear_range=0.05,
                            zoom_range=0.05,
                            horizontal_flip=True,
                            fill_mode='nearest')

train_gen = train_generator(BATCH_SIZE,
                            root_path+'train',
                            'IMG',
                            'mask',
                            train_generator_args,
                            target_size=(512,512))

val_gen = val_generator(BATCH_SIZE,
                            root_path+'val',
                            'IMG',
                            'mask',
                            train_generator_args,
                            target_size=(512,512))

# validation_data = (test_load_image(data_path+val_list[0], target_size=(512, 512)),
#                     test_load_image(mask_path+val_list[0], target_size=(512, 512)))

model_checkpoint = ModelCheckpoint('unet_lung_seg.hdf5',
                                   monitor='loss',
                                   verbose=1,
                                   save_best_only=True)

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

training_loss = history.history['loss']
validation_loss = history.history['val_loss']

training_accuracy = history.history['binary_accuracy']
validation_accuracy = history.history['val_binary_accuracy']

epoch_count = range(1, len(training_loss) + 1)

axs[0].plot(epoch_count, training_loss, 'r--')
axs[0].plot(epoch_count, validation_loss, 'b-')
axs[0].legend(['Training Loss', 'Validation Loss'])


axs[1].plot(epoch_count, training_accuracy, 'r--')
axs[1].plot(epoch_count, validation_accuracy, 'b-')
axs[1].legend(['Training Accuracy', 'Validation Accuracy'])
plt.savefig("Traing_Error_Accuracy")
# plt.show()

LossFile = fopen('Loss.txt', "w", encoding = "utf8")
for i in range(len(epoch_count)):
    LossFile.writelines([train_loss[i], validation_loss[i]])
LossFile.close()

AccFile = fopen('Accuracy.txt', "w", encoding = "utf8")
for i in range(len(epoch_count)):
    AccFile.writelines([training_accuracy[i], validation_accuracy[i]])
AccFile.close()

