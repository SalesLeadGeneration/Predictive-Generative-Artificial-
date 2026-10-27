from keras import Sequential
import numpy as np
from keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten, Dense
from Classificaltion_Evaluation import ClassificationEvaluation


def Model_CNN(train_data, train_target, test_data, test_target, Hidden):
    print('CNN')
    IMG_SIZE = 28
    Train_X = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 1))
    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (IMG_SIZE * IMG_SIZE, 1))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 1))

    Test_X = np.zeros((test_data.shape[0], IMG_SIZE, IMG_SIZE, 1))
    for i in range(test_data.shape[0]):
        temp = np.resize(test_data[i], (IMG_SIZE * IMG_SIZE, 1))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 1))
    X_train = Train_X.astype('float32') / 255
    X_test = Test_X.astype('float32') / 255
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(Hidden, activation='relu'),
        Dropout(0.5),
        Dense(train_target.shape[1], activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, train_target, epochs=15, batch_size=64, validation_split=0.1)
    pred = model.predict(X_test)

    min = np.min(pred)
    max = np.max(pred)
    avg = (min + max) / 2
    pred[pred >= avg] = 1
    pred[pred < avg] = 0

    Eval = ClassificationEvaluation(test_target, pred)
    return Eval

