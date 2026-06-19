import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    Reshape
)

def build_bilstm(input_dim, num_classes):

    model = Sequential()

    model.add(
        Reshape(
            (input_dim, 1),
            input_shape=(input_dim,)
        )
    )

    model.add(
        Bidirectional(
            LSTM(
                64,
                return_sequences=False
            )
        )
    )

    model.add(
        Dropout(0.3)
    )

    model.add(
        Dense(
            64,
            activation='relu'
        )
    )

    model.add(
        Dense(
            num_classes,
            activation='softmax'
        )
    )

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
