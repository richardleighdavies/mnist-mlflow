""" """

import sys

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)

from tensorflow.keras.optimizers import Adam

import metrics

def Conv2D_ReLU_BN(input_function, filters, batch_norm=True):

    x = Conv2D(filters, (3, 3), activation='relu')(input_function)

    if batch_norm:
        x = BatchNormalization()(x)

    return x


def Dense_ReLU_BN(input_function, units, batch_norm=False):

    x = Dense(units, activation='relu')(input_function)

    if batch_norm:
        x = BatchNormalization()(x)

    return x


def ConvNet(input_shape, output_classes, batch_norm=True):

    inputs = Input(input_shape)

    batch_norm_01 = BatchNormalization()(inputs)

    conv_01 = Conv2D_ReLU_BN(batch_norm_01 if batch_norm else inputs, filters=64, batch_norm=batch_norm)
    conv_02 = Conv2D_ReLU_BN(conv_01, filters=64, batch_norm=batch_norm)

    max_pool_01 = MaxPooling2D((2, 2))(conv_02)

    conv_03 = Conv2D_ReLU_BN(max_pool_01, filters=128, batch_norm=batch_norm)
    conv_04 = Conv2D_ReLU_BN(conv_03, filters=128, batch_norm=batch_norm)

    max_pool_02 = MaxPooling2D((2, 2))(conv_04)

    flatten = Flatten()(max_pool_02)

    dense_01 = Dense_ReLU_BN(flatten, units=128, batch_norm=batch_norm)
    dense_02 = Dense_ReLU_BN(dense_01, units=128, batch_norm=batch_norm)

    outputs = Dense(output_classes, 'softmax')(dense_02)

    model = Model(inputs, outputs)

    return model

def build(params):

    model = ConvNet(
        input_shape=(28, 28, 1),
        output_classes=10,
        batch_norm=params.batch_norm
    )

    if params.model_summary:
        model.summary()
        sys.exit()

    model.compile(
        optimizer=Adam(learning_rate=params.learning_rate),
        loss='categorical_crossentropy',
        metrics=metrics.fetch()
    )

    if params.load_weights:
        model.load_weights(params.load_weights)

    return model
