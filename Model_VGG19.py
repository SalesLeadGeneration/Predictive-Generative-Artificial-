import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from Classificaltion_Evaluation import ClassificationEvaluation


def VGG_19(num_of_class=None, hidden=None):
    model = Sequential()

    # Block 1
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=(32, 32, 3)))
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 2
    model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 3
    model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 4
    model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 5
    model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(4096, activation="relu"))
    if hidden:
        model.add(Dense(hidden, activation="relu"))
    model.add(Dense(num_of_class, activation="softmax"))

    return model


def Model_VGG19(Train_Data, Train_Tar, Test_Data, Test_Tar, Hidden_Neuron_Count):
    IMG_SIZE = 32
    # Preprocess training data
    Train_X = np.zeros((Train_Data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Train_Data.shape[0]):
        Train_X[i] = np.resize(Train_Data[i], (IMG_SIZE, IMG_SIZE, 3))

    # Preprocess testing data
    Test_X = np.zeros((Test_Data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Test_Data.shape[0]):
        Test_X[i] = np.resize(Test_Data[i], (IMG_SIZE, IMG_SIZE, 3))

    # Build model
    model = VGG_19(num_of_class=Train_Tar.shape[1], hidden=Hidden_Neuron_Count)
    model.summary()
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # Train model
    model.fit(x=Train_X, y=Train_Tar, epochs=5, steps_per_epoch=10)

    # Predict
    pred = model.predict(Test_X).astype('int')
    min_val = np.min(pred)
    max_val = np.max(pred)
    avg = (min_val + max_val) / 2
    pred[pred >= avg] = 1
    pred[pred < avg] = 0

    # Evaluate
    Eval = ClassificationEvaluation(Test_Tar, pred)
    return np.asarray(Eval).ravel()
