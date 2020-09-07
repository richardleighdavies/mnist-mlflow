""" """

from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall
from tensorflow_addons.metrics import F1Score

def fetch(classes=10):

    metrics = [
        CategoricalAccuracy(),
        Precision(),
        Recall(),
        F1Score(num_classes=classes, average='micro', name='micro_f1_score'),
        F1Score(num_classes=classes, average='macro', name='macro_f1_score'),
    ]

    return metrics
