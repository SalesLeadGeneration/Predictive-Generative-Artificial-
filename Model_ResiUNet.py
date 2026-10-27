from keras import Input, Model
from keras.layers import Conv2D, Add, Activation, BatchNormalization
from keras.src.layers import MaxPooling2D, Conv2DTranspose
from numpy import concatenate
import numpy as np
import random as rn
import cv2 as cv

from Evaluation_All import seg_evaluation


def residual_block(x, filters, kernel_size=3):
    conv = Conv2D(filters, kernel_size, padding='same')(x)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    conv = Conv2D(filters, kernel_size, padding='same')(conv)
    conv = BatchNormalization()(conv)

    shortcut = Conv2D(filters, 1, padding='same')(x)
    shortcut = BatchNormalization()(shortcut)

    res = Add()([conv, shortcut])
    res = Activation('relu')(res)
    return res


def resiunet_model(input_shape=(256, 256, 3)):
    inputs = Input(shape=input_shape)

    # Encoder
    conv1 = residual_block(inputs, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = residual_block(pool1, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = residual_block(pool2, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = residual_block(pool3, 512)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = residual_block(pool4, 1024)

    # Decoder
    up6 = Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(conv5)
    up6 = concatenate([up6, conv4], axis=3)
    conv6 = residual_block(up6, 512)

    up7 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv3], axis=3)
    conv7 = residual_block(up7, 256)

    up8 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
    up8 = concatenate([up8, conv2], axis=3)
    conv8 = residual_block(up8, 128)

    up9 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
    up9 = concatenate([up9, conv1], axis=3)
    conv9 = residual_block(up9, 64)

    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def Model_ResiUNet(Images, Target):
    # Set parameters
    IMG_SIZE = 256
    seed = 42
    rn.seed(seed)
    np.random.seed(seed)

    X_train = np.zeros((Images.shape[0], IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    Y_train = np.zeros((Target.shape[0], IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    for i in range(Images.shape[0]):
        Temp = cv.resize(Images[i, :], (IMG_SIZE, IMG_SIZE))
        X_train[i, :, :, :] = Temp.reshape((IMG_SIZE, IMG_SIZE, 3))

    for i in range(Target.shape[0]):
        Temp = cv.resize(Target[i, :], (IMG_SIZE, IMG_SIZE))
        Temp = Temp.reshape((IMG_SIZE, IMG_SIZE, 3))
        Temp = np.where(Temp < 0.5, 0, 1)  # Binarize mask
        Y_train[i, :, :, :] = Temp

    model = resiunet_model()
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=10, batch_size=8, verbose=1)

    pred_img = model.predict(X_train)
    ret_img = pred_img[:, :, :, 0]
    Eval = seg_evaluation(ret_img, Target)
    return Eval, ret_img
