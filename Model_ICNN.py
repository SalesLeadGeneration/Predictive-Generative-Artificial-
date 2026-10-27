from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation
from keras.layers import AveragePooling2D
import numpy as np
import random as rn
import cv2 as cv

from Evaluation_All import seg_evaluation


def inception_block(x, filters):
    # 1x1 conv
    conv1 = Conv2D(filters, 1, padding='same', activation='relu')(x)

    # 3x3 conv
    conv3 = Conv2D(filters, 1, padding='same', activation='relu')(x)
    conv3 = Conv2D(filters, 3, padding='same', activation='relu')(conv3)

    # 5x5 conv
    conv5 = Conv2D(filters, 1, padding='same', activation='relu')(x)
    conv5 = Conv2D(filters, 5, padding='same', activation='relu')(conv5)

    # 3x3 average pooling
    pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    pool = Conv2D(filters, 1, padding='same', activation='relu')(pool)

    # Concatenate all
    out = concatenate([conv1, conv3, conv5, pool], axis=-1)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out


def icnn_model(input_shape=(256, 256, 3)):
    inputs = Input(shape=input_shape)

    # Encoder
    conv1 = inception_block(inputs, 32)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = inception_block(pool1, 64)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = inception_block(pool2, 128)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = inception_block(pool3, 256)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = inception_block(pool4, 512)

    # Decoder
    up6 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv5)
    up6 = concatenate([up6, conv4], axis=-1)
    conv6 = inception_block(up6, 256)

    up7 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv3], axis=-1)
    conv7 = inception_block(up7, 128)

    up8 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv7)
    up8 = concatenate([up8, conv2], axis=-1)
    conv8 = inception_block(up8, 64)

    up9 = Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(conv8)
    up9 = concatenate([up9, conv1], axis=-1)
    conv9 = inception_block(up9, 32)

    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def Model_ICNN(Images, Target):
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

    model = icnn_model()
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=10, batch_size=8, verbose=1)

    pred_img = model.predict(X_train)
    ret_img = pred_img[:, :, :, 0]
    Eval = seg_evaluation(ret_img, Target)
    return Eval, ret_img
