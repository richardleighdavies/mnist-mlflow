""" """

from functools import partial

from tensorflow.data import Dataset
from tensorflow.data.experimental import AUTOTUNE

from augmentations import initialize_augmentations, tf_augmentations

import data


def generator(params, x, y, instances, augment=True):

    if augment:
        initialize_augmentations(params)

    dataset = Dataset.from_tensor_slices((x, y))

    dataset = dataset.shuffle(instances)

    dataset = dataset.repeat()

    dataset = dataset.batch(params.batch_size, drop_remainder=True)

    if augment:
        dataset = dataset.map(partial(tf_augmentations), num_parallel_calls=AUTOTUNE, deterministic=False)

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


def create(params):

    (x_train, y_train), (x_test, y_test) = data.load(params)

    training_generator = generator(params, x_train, y_train, params.training_instances, augment=params.augment)
    validation_generator = generator(params, x_test, y_test, params.validation_instances, augment=False)

    return training_generator, validation_generator
