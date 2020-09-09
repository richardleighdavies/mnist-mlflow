''' '''

import json

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

import mlflow

from clr import CyclicLearningRate


class MLFlowCallback(Callback):

    def __init__(self, params):

        self.parameters = params

        if params.model_checkpoint_mode == 'min':
            self.model_checkpoint_best_value = float('inf')
        else:
            self.model_checkpoint_best_value = float('-inf')

    def on_train_begin(self, logs=None):

        self.log_parameters()

        self.log_model_architecture()

    def on_train_end(self, logs=None):

        self.log_model()

    def on_epoch_end(self, epoch, logs=None):

        for key, value in logs.items():
            mlflow.log_metric(key=key, value=value, step=epoch)

        learning_rate = K.eval(self.model.optimizer.lr)

        mlflow.log_metric(key='learning_rate', value=learning_rate, step=epoch)

        self.model_checkpoint_check(epoch, logs)

    def log_parameters(self):

        for key, value in vars(self.parameters).items():
            mlflow.log_param(key=key, value=value)

    def log_model_architecture(self):

        json_object = json.loads(self.model.to_json())

        json_string = json.dumps(json_object, indent=4)

        with open('./artifacts/model_architecture.json', 'w') as file:
            file.write(json_string)

        mlflow.log_artifact('./artifacts/model_architecture.json')

    def model_checkpoint_check(self, epoch, logs=None):

        current_value = logs[self.params.model_checkpoint_monitor]

        if self.model_checkpoint_mode == 'min':
            if current_value < self.model_checkpoint_best_value:
                self.log_model()

        if self.model_checkpoint_mode == 'max':
            if current_value > self.model_checkpoint_best_value:
                self.log_model()

    def log_model(self):

        input_example = np.zeros((28, 28, 1), np.uint8)

        mlflow.keras.log_model(
            keras_model=self.model,
            artifact_path='models',
            conda_env='./conda.yaml',
            custom_objects=None,
            keras_module=tf.keras,
            input_example=input_example,

        )

        self.print_model_checkpoint_message()

    def model_checkpoint_message(self):

        if self.params.model_checkpoint_verbose == 1:
            print(f'')

        elif self.params.model_checkpoint_verbose == 2:
            print(f'')


def get(params):

    callbacks = [
        ModelCheckpoint(
            filepath='./models/model.h5',
            monitor='val_micro_f1_score',
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode='max',
        ),
        MLFlowCallback(params),
    ]

    if params.cyclic_learning_rate:
        callbacks += [
            CyclicLearningRate(
                base_lr=params.cyclic_learning_rate_min,
                max_lr=params.cyclic_learning_rate_max,
                step_size=params.training_step_size * params.cyclic_learning_rate_step,
            )
        ]

    return callbacks
