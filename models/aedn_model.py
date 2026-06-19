import tensorflow as tf

from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    MultiHeadAttention,
    LayerNormalization,
    Add,
    Reshape,
    Flatten
)

def build_aedn(input_dim, num_classes):

    inputs = Input(shape=(input_dim,))

    x = Dense(
        128,
        activation='relu'
    )(inputs)

    x = Dropout(0.30)(x)

    x = Dense(
        64,
        activation='relu'
    )(x)

    x = Dropout(0.30)(x)

    x_seq = Reshape((64, 1))(x)

    attention_output = MultiHeadAttention(
        num_heads=4,
        key_dim=16
    )(x_seq, x_seq)

    x_seq = Add()(
        [x_seq, attention_output]
    )

    x_seq = LayerNormalization()(
        x_seq
    )

    x = Flatten()(x_seq)

    x = Dense(
        32,
        activation='relu'
    )(x)

    x = Dropout(0.30)(x)

    outputs = Dense(
        num_classes,
        activation='softmax'
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001
    )

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
