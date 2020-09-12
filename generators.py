""" """

from functools import partial

from tensorflow.data import Dataset
from tensorflow.data.experimental import AUTOTUNE

from augmentations import Augmentations


def create(params, x, y, instances, augment=False):

    if augment:
        augmentations = Augmentations(params)

    dataset = Dataset.from_tensor_slices((x, y))

    dataset = dataset.shuffle(instances)

    dataset = dataset.repeat()

    dataset = dataset.batch(params.batch_size, drop_remainder=True)

    if augment:
        dataset = dataset.map(partial(augmentations.tf_augmentations), num_parallel_calls=AUTOTUNE, deterministic=False)

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset
