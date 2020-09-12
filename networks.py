""" """

import sys

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)

from tensorflow.keras.optimizers import Adam

import metrics


def Conv2D_ReLU_BN(input_function, filters):

    x = Conv2D(filters, (3, 3), activation='relu')(input_function)

    x = BatchNormalization()(x)


    return x


def Dense_ReLU_BN(input_function, units, dropout=False, dropout_rate=0.2):

    x = Dense(units, activation='relu')(input_function)

    if dropout:
        x = Dropout(dropout_rate)(x)

    x = BatchNormalization()(x)

    return x


def ConvNet(input_shape, output_classes, dropout=False, dropout_rate=0.5):

    inputs = Input(input_shape)

    batch_norm_01 = BatchNormalization()(inputs)

    conv_01 = Conv2D_ReLU_BN(batch_norm_01, filters=64)
    conv_02 = Conv2D_ReLU_BN(conv_01, filters=64)

    max_pool_01 = MaxPooling2D((2, 2))(conv_02)

    conv_03 = Conv2D_ReLU_BN(max_pool_01, filters=128)
    conv_04 = Conv2D_ReLU_BN(conv_03, filters=128)

    max_pool_02 = MaxPooling2D((2, 2))(conv_04)

    flatten = Flatten()(max_pool_02)

    dense_01 = Dense_ReLU_BN(flatten, units=128, dropout=dropout, dropout_rate=dropout_rate)
    dense_02 = Dense_ReLU_BN(dense_01, units=128, dropout=dropout, dropout_rate=dropout_rate)

    outputs = Dense(output_classes, 'softmax')(dense_02)

    model = Model(inputs, outputs)

    return model


def build(params):

    model = ConvNet(
        input_shape=(28, 28, 1),
        output_classes=10,
        dropout=params.dropout,
        dropout_rate=params.dropout_rate,
    )

    if params.model_summary:
        model.summary()
        sys.exit()

    model.compile(
        optimizer=Adam(learning_rate=params.learning_rate),
        loss='categorical_crossentropy',
        metrics=metrics.fetch()
    )

    if params.load_weights != 'None':
        model.load_weights(params.load_weights)

    return model
