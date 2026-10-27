import numpy as np
from sklearn.preprocessing import OneHotEncoder
from Classificaltion_Evaluation import ClassificationEvaluation
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Multiply, Softmax, Lambda
import tensorflow as tf


def attention_block(inputs):
    """
    Simple attention mechanism for a fully connected DNN.
    Computes attention weights for each feature.
    """
    attention_scores = Dense(inputs.shape[-1], activation='tanh')(inputs)
    attention_scores = Dense(inputs.shape[-1], activation='softmax')(attention_scores)

    attended = Multiply()([inputs, attention_scores])
    return attended


def Model_Attention_guidedDNN(data, labels, test_data, test_target, sol=None):
    if sol is None:
        sol = [64, 0.01]

    if len(data.shape) == 3:
        data = data.reshape(data.shape[0], -1)
        test_data = test_data.reshape(test_data.shape[0], -1)

    num_classes = 23

    if len(labels.shape) == 1:
        encoder = OneHotEncoder(sparse=False)
        labels = encoder.fit_transform(labels.reshape(-1, 1))

    inputs = Input(shape=(data.shape[1],))

    x = Dense(sol[0], activation='relu')(inputs)

    x = attention_block(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=sol[1]),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(data, labels, epochs=5, batch_size=32, verbose=1)

    pred = model.predict(test_data)
    Pred = np.argmax(pred, axis=1)
    if len(test_target.shape) > 1 and test_target.shape[1] > 1:
        test_target_labels = np.argmax(test_target, axis=1)
    else:
        test_target_labels = test_target

    Eval = ClassificationEvaluation(test_target_labels, Pred)
    return Eval
