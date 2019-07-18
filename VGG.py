from __future__ import print_function
from __future__ import absolute_import

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose, UpSampling2D
from keras.layers import  BatchNormalization, Activation, average
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
'''
Taken from https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3 on April 20, 2019
'''
def VGG_16qua(weights_path=None, size_set = 512):

    img_input = Input(shape=(size_set, size_set, 7))

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Block 6
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='softmax', name='predictions')(x)

    model = Model(inputs=[img_input], outputs=[x])

    # model = Sequential()
    # model.add(ZeroPadding2D((1,1),input_shape=(size_set,size_set,7)))
    # model.add(Convolution2D(64, 3, 3, activation='relu'))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(64, 3, 3, activation='relu'))
    # model.add(MaxPooling2D((2,2), strides=(2,2)))
    #
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(128, 3, 3, activation='relu'))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(128, 3, 3, activation='relu'))
    # model.add(MaxPooling2D((2,2), strides=(2,2)))
    #
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(256, 3, 3, activation='relu'))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(256, 3, 3, activation='relu'))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(256, 3, 3, activation='relu'))
    # model.add(MaxPooling2D((2,2), strides=(2,2)))
    #
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu'))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu'))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu'))
    # model.add(MaxPooling2D((2,2), strides=(2,2)))
    #
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu'))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu'))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu'))
    # model.add(MaxPooling2D((2,2), strides=(2,2)))
    #
    # model.add(Flatten())
    # model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1, activation='softmax'))
    #
    # if weights_path:
    #     model.load_weights(weights_path)

    return model
