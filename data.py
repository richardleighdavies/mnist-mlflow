""" """

from numpy import ceil as np_ceil

from tensorflow.keras.datasets import mnist

from tensorflow.keras.utils import to_categorical


def preprocess(x_train, y_train, x_test, y_test):

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    y_train = to_categorical(y_train, dtype='uint8')
    y_test = to_categorical(y_test, dtype='uint8')

    return (x_train, y_train), (x_test, y_test)


def load(params):

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    (x_train, y_train), (x_test, y_test) = preprocess(x_train, y_train, x_test, y_test)

    params.training_instances = x_train.shape[0]
    params.validation_instances = x_test.shape[0]

    params.training_step_size = int(np_ceil(params.training_instances / params.batch_size))
    params.validation_step_size = int(np_ceil(params.validation_instances / params.batch_size))

    return (x_train, y_train), (x_test, y_test)
