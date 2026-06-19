import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Dense,
    Dropout,
    Flatten,
    Reshape
)

def build_cnn(input_dim, num_classes):

    model = Sequential()

    model.add(
        Reshape(
            (input_dim, 1),
            input_shape=(input_dim,)
        )
    )

    model.add(
        Conv1D(
            filters=64,
            kernel_size=3,
            activation='relu',
            padding='same'
        )
    )

    model.add(
        MaxPooling1D(pool_size=2)
    )

    model.add(
        Conv1D(
            filters=128,
            kernel_size=3,
            activation='relu',
            padding='same'
        )
    )

    model.add(
        MaxPooling1D(pool_size=2)
    )

    model.add(Flatten())

    model.add(
        Dense(
            128,
            activation='relu'
        )
    )

    model.add(
        Dropout(0.3)
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
