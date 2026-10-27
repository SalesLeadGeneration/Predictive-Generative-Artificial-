import numpy as np
import cv2 as cv
from keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from Classificaltion_Evaluation import ClassificationEvaluation

def Model_Resnet(train_data, train_target, test_data, test_target, sol=None):
    if sol is None:
        sol = [64, 5]  # sol[0]=hidden neurons, sol[1]=epochs

    IMG_SIZE = [224, 224, 3]

    # Preprocess training data
    Feat_train = np.zeros((train_data.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    for i in range(train_data.shape[0]):
        Temp = cv.resize(train_data[i], (IMG_SIZE[1], IMG_SIZE[0]))
        if Temp.shape[-1] == 1:
            Temp = np.repeat(Temp, 3, axis=2)
        Feat_train[i] = Temp
    X_train = Feat_train / 255.0  # normalize

    # Preprocess test data
    Feat_test = np.zeros((test_data.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    for i in range(test_data.shape[0]):
        Temp = cv.resize(test_data[i], (IMG_SIZE[1], IMG_SIZE[0]))
        if Temp.shape[-1] == 1:
            Temp = np.repeat(Temp, 3, axis=2)
        Feat_test[i] = Temp
    X_test = Feat_test / 255.0

    # Build model
    base_model = Sequential()
    base_model.add(ResNet50(include_top=False, weights='imagenet', pooling='avg', input_shape=(224,224,3)))
    base_model.add(Dense(units=sol[0], activation='relu'))
    base_model.add(Dense(units=train_target.shape[1], activation='sigmoid'))  # final output layer

    base_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    base_model.fit(X_train, train_target, epochs=sol[1], batch_size=8, verbose=1)

    # Predict on test data
    pred_test = np.round(base_model.predict(X_test)).astype('int')
    Eval_test = ClassificationEvaluation(test_target, pred_test)

    return Eval_test
