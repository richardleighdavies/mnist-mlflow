''' '''

import numpy as np

from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K

VALID_MODES = ('triangular', 'triangular2', 'exp_range')


class CyclicLearningRate(Callback):
    """ """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000,
            mode='triangular', gamma=1.0):

        if mode not in VALID_MODES:
            raise Exception(f'mode must be one of: {VALID_MODES}')

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size

        if mode == 'triangular':
            self.scale_fn = lambda x: 1.0
            self.scale_mode = 'cycle'

        elif mode == 'triangular2':
            self.scale_fn = lambda x: 1.0 / 2.0 ** (x - 1.0)
            self.scale_mode = 'cycle'

        else:
            self.scale_fn = lambda x: gamma ** x
            self.scale_mode = 'iterations'

        self.step = 0

    def calculate_learning_rate(self):

        cycle = np.floor(1 + self.step / (2 * self.step_size))

        w = np.abs(self.step / self.step_size - 2 * cycle + 1)
        x = self.max_lr - self.base_lr
        y = np.maximum(0, 1 - w)
        z = self.scale_fn(cycle if self.scale_mode == 'cycle' else self.step)

        return self.base_lr + x * y * z

    def on_train_begin(self, logs=None):

        K.set_value(self.model.optimizer.lr, self.base_lr)

    def on_batch_end(self, batch, logs=None):

        learning_rate = self.calculate_learning_rate()

        K.set_value(self.model.optimizer.lr, learning_rate)

        self.step += 1
