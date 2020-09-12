""" """

from numpy import ceil as np_ceil

from tensorflow.keras.datasets import mnist

from tensorflow.keras.utils import to_categorical

import preprocessing


def load_mnist(params):

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    (x_train, y_train), (x_test, y_test) = preprocess_mnist(x_train, y_train, x_test, y_test)

    params.training_instances = x_train.shape[0]
    params.validation_instances = x_test.shape[0]

    params.training_step_size = int(np_ceil(params.training_instances / params.batch_size))
    params.validation_step_size = int(np_ceil(params.validation_instances / params.batch_size))

    return (x_train, y_train), (x_test, y_test)
