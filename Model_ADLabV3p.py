import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, concatenate, AveragePooling2D
import numpy as np
import cv2 as cv
import random as rn
from Evaluation_All import seg_evaluation


def ASPP(x, out_channels=256):
    dims = x.shape

    # 1x1 conv
    conv1 = Conv2D(out_channels, 1, padding='same', use_bias=False)(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    # 3x3 conv rates 6, 12, 18
    conv3_1 = Conv2D(out_channels, 3, padding='same', dilation_rate=6, use_bias=False)(x)
    conv3_1 = BatchNormalization()(conv3_1)
    conv3_1 = Activation('relu')(conv3_1)

    conv3_2 = Conv2D(out_channels, 3, padding='same', dilation_rate=12, use_bias=False)(x)
    conv3_2 = BatchNormalization()(conv3_2)
    conv3_2 = Activation('relu')(conv3_2)

    conv3_3 = Conv2D(out_channels, 3, padding='same', dilation_rate=18, use_bias=False)(x)
    conv3_3 = BatchNormalization()(conv3_3)
    conv3_3 = Activation('relu')(conv3_3)

    # Image pooling
    pool = AveragePooling2D(pool_size=(dims[1], dims[2]))(x)
    pool = Conv2D(out_channels, 1, padding='same', use_bias=False)(pool)
    pool = BatchNormalization()(pool)
    pool = Activation('relu')(pool)
    pool = UpSampling2D(size=(dims[1], dims[2]), interpolation='bilinear')(pool)

    # Concatenate
    x = concatenate([conv1, conv3_1, conv3_2, conv3_3, pool])
    x = Conv2D(out_channels, 1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def ADLabV3Plus(input_shape=(256, 256, 3), num_classes=1, backbone='mobilenetv2'):
    inputs = Input(shape=input_shape)

    if backbone == 'mobilenetv2':
        base_model = tf.keras.applications.MobileNetV2(input_tensor=inputs,
                                                       include_top=False,
                                                       weights='imagenet')
        low_level_feat = base_model.get_layer('block_3_expand_relu').output
        x = base_model.get_layer('out_relu').output
    else:
        raise ValueError("Only 'mobilenetv2' backbone implemented.")

    # ASPP
    x = ASPP(x, out_channels=256)
    x = UpSampling2D(size=(4, 4), interpolation='bilinear')(x)  # Adaptive upsample

    # Decoder with low-level features
    low_level_feat = Conv2D(48, 1, padding='same', use_bias=False)(low_level_feat)
    low_level_feat = BatchNormalization()(low_level_feat)
    low_level_feat = Activation('relu')(low_level_feat)

    x = concatenate([x, low_level_feat])
    x = Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D(size=(4, 4), interpolation='bilinear')(x)  # Final upsample
    outputs = Conv2D(num_classes, 1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def Model_ADLabV3p(Images, Target, sol=None):
    if sol is None:
        sol = [256, 0.001, 100]

    IMG_SIZE = 256
    seed = 42
    rn.seed(seed)
    np.random.seed(seed)

    # Preprocess images
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

    # Build model
    model = ADLabV3Plus(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=1)

    # Update convolution filters dynamically based on sol[0]
    for layer in model.layers:
        if isinstance(layer, Conv2D) and layer.filters != 1:  # skip output layer
            layer.filters = sol[0]

    # Compile with learning rate from sol[1]
    optimizer = tf.keras.optimizers.Adam(learning_rate=sol[1])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    # Fit model using sol[2] as steps_per_epoch
    model.fit(X_train, Y_train, epochs=10, batch_size=8, verbose=1, steps_per_epoch=sol[2])

    # Predict & evaluate
    pred_img = model.predict(X_train)
    ret_img = pred_img[:, :, :, 0]
    Eval = seg_evaluation(ret_img, Target)
    return Eval, ret_img

