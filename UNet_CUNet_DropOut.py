from __future__ import print_function
from __future__ import absolute_import

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose, UpSampling2D
from keras.layers import  BatchNormalization, Activation, average

def DeepModel(size_set = 256):

    img_input = Input(shape=(size_set, size_set, 1))

    conv1 = Conv2D(32, (3, 3), padding='same', activation='relu',name='layer')(img_input)
    conv1 = Conv2D(32, (3, 3), padding='same', activation='relu',name='block1_conv2')(conv1)
    conv1_dir = Conv2D(32, (3, 3), padding='same', activation='relu',name='block1_conv2_dir')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), padding='same', activation='relu',name='block2_conv1')(pool1)
    conv2 = Conv2D(64, (3, 3), padding='same', activation='relu',name='block2_conv2')(conv2)
    conv2_dir = Conv2D(64, (3, 3), padding='same', activation='relu',name='block2_conv2_dir')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), padding='same', activation='relu',name='block3_conv1')(pool2)
    conv3 = Conv2D(128, (3, 3), padding='same', activation='relu',name='block3_conv2')(conv3)
    conv3_dir = Conv2D(128, (3, 3), padding='same', activation='relu',name='block3_conv2_dir')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    drop3 = Dropout(0.5)(pool3, training=True)

    conv4 = Conv2D(256, (3, 3), padding='same', activation='relu',name='block4_conv1')(drop3)
    conv4 = Conv2D(256, (3, 3), padding='same', activation='relu',name='block4_conv2')(conv4)
    conv4_dir = Conv2D(256, (3, 3), padding='same', activation='relu',name='block4_conv2_dir')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    drop4 = Dropout(0.5)(pool4, training=True)

    conv5 = Conv2D(512, (3, 3), padding='same', activation='relu',name='block5_conv1')(drop4)
    conv5 = Conv2D(512, (3, 3), padding='same', activation='relu',name='block5_conv2')(conv5)

    drop5 = Dropout(0.5)(conv5, training=True)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', name='block6_dconv')(drop5), conv4_dir], axis=3)
    conv6 = Conv2D(256, (3, 3), padding='same', activation='relu',name='block6_conv1')(up6)
    conv6 = Conv2D(256, (3, 3), padding='same', activation='relu',name='block6_conv2')(conv6)

    drop6 = Dropout(0.5)(conv6, training=True)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name='block7_dconv')(drop6), conv3_dir], axis=3)
    conv7 = Conv2D(128, (3, 3), padding='same', activation='relu',name='block7_conv1')(up7)
    conv7 = Conv2D(128, (3, 3), padding='same', activation='relu',name='block7_conv2')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name='block8_dconv')(conv7), conv2_dir], axis=3)
    conv8 = Conv2D(64, (3, 3), padding='same', activation='relu',name='block8_conv1')(up8)
    conv8 = Conv2D(64, (3, 3), padding='same', activation='relu', name='block8_conv2')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', name='block9_dconv')(conv8), conv1_dir], axis=3)
    conv9 = Conv2D(32, (3, 3), padding='same', activation='relu', name='block9_conv1')(up9)
    conv9 = Conv2D(32, (3, 3), padding='same', activation='relu', name='block9_conv2')(conv9)

    conv10 = Conv2D(6, (1,1), activation='sigmoid')(conv9)

    model = Model(inputs=[img_input], outputs=[conv10])

    return model
