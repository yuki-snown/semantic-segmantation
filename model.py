from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Activation
from keras.models import Model

def segNet_layer(x, channels):
    x = Conv2D(channels, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def segNet_layer_2d(x, channels):
    x = segNet_layer(x, channels)
    x = segNet_layer(x, channels)
    return x

def segNet_layer_3d(x, channels):
    x = segNet_layer(x, channels)
    x = segNet_layer(x, channels)
    x = segNet_layer(x, channels)
    return x

def SegNet(input_size=(480, 480, 3), class_num=2): # hw: 32h, 32w
    input_shape = Input(shape=input_size)

    x = segNet_layer_2d(input_shape, 32)
    x = MaxPooling2D()(x)

    x = segNet_layer_2d(x, 64)
    x = MaxPooling2D()(x)

    x = segNet_layer_3d(x, 128)
    x = MaxPooling2D()(x)
    
    x = segNet_layer_3d(x, 256)
    x = MaxPooling2D()(x)
    
    x = segNet_layer_3d(x, 512)
    x = MaxPooling2D()(x)
    
    x = UpSampling2D()(x)
    x = segNet_layer_3d(x, 512)

    x = UpSampling2D()(x)
    x = segNet_layer_3d(x, 256)
    
    x = UpSampling2D()(x)
    x = segNet_layer_3d(x, 128)
    
    x = UpSampling2D()(x)
    x = segNet_layer_2d(x, 64)

    x = UpSampling2D()(x)
    x = segNet_layer_2d(x, 32)
    x = Conv2D(class_num, (1,1), padding='same', activation='softmax')(x)

    return Model(inputs=input_shape, outputs=x)


def Unet(input_size=(480, 480, 3), class_num=2): # hw: 32h, 32w
    input_shape = Input(shape=input_size)

    x1 = Conv2D(64, (3,3), padding='same', activation='relu')(input_shape)
    x1 = Conv2D(64, (3,3), padding='same', activation='relu')(x1)
    x2 = MaxPooling2D()(x1)

    x2 = Conv2D(128, (3,3), padding='same', activation='relu')(x2)
    x2 = Conv2D(128, (3,3), padding='same', activation='relu')(x2)
    x3 = MaxPooling2D()(x2)

    x3 = Conv2D(256, (3,3), padding='same', activation='relu')(x3)
    x3 = Conv2D(256, (3,3), padding='same', activation='relu')(x3)
    x4 = MaxPooling2D()(x3)

    x4 = Conv2D(512, (3,3), padding='same', activation='relu')(x4)
    x4 = Conv2D(512, (3,3), padding='same', activation='relu')(x4)
    x5 = MaxPooling2D()(x4)

    x5 = Conv2D(1024, (3,3), padding='same', activation='relu')(x5)
    x5 = Conv2D(512, (3,3), padding='same', activation='relu')(x5)

    x = UpSampling2D()(x5)
    x = Concatenate()([x, x4])
    x = Conv2D(512, (3,3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3,3), padding='same', activation='relu')(x)

    x = UpSampling2D()(x)
    x = Concatenate()([x, x3])
    x = Conv2D(256, (3,3), padding='same', activation='relu')(x)
    x = Conv2D(128, (3,3), padding='same', activation='relu')(x)

    x = UpSampling2D()(x)
    x = Concatenate()([x, x2])
    x = Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3,3), padding='same', activation='relu')(x)

    x = UpSampling2D()(x)
    x = Concatenate()([x, x1])
    
    x = Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = Conv2D(class_num, (1,1), padding='same', activation='softmax')(x)

    return Model(inputs=input_shape, outputs=x)
