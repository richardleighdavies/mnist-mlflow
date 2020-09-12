""" """

from tensorflow import numpy_function

from albumentations import (
    Compose, HorizontalFlip, RGBShift, RandomBrightnessContrast, ShiftScaleRotate
)

class Augmentations:

    def __init__(self, params):

        self.initialize_augmentations(params)

    def initialize_augmentations(self):

        transformations = [
            ShiftScaleRotate(
                shift_limit=params.translation,
                scale_limit=params.scale,
                rotate_limit=params.rotation,
                interpolation=0, border_mode=0, value=0, always_apply=True),
            # RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
            RandomBrightnessContrast(
                brightness_limit=params.brightness,
                contrast_limit=0.0,
                always_apply=True),
        ]

        if params.horizontal_flip:
            transformations += [ HorizontalFlip() ]

        if params.vertical_flip:
            transformations += [ VerticalFlip() ]

        self.transformations = Compose(transformations)

    def tf_augmentations(self, x, y):

        x = numpy_function(func=numpy_augmentations, inp=[x], Tout='uint8')

        return x, y

    def numpy_augmentations(self, x):

        for i in range(x.shape[0]):
            x[i] = self.transformations(image=x[i])['image']

        return x
