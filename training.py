#!/usr/bin/python3
''' '''

import mlflow

import warnings
warnings.simplefilter('ignore')

import callbacks
import generators
import network

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('run_name', type=str)

parser.add_argument('--load_weights', type=str)

parser.add_argument('--batch_size', type=int)
parser.add_argument('--epochs', type=int)

parser.add_argument('--dropout', type=str, choices=['True', 'False'])
parser.add_argument('--dropout_rate', type=float)

parser.add_argument('--learning_rate', type=float)

parser.add_argument('--cyclic_learning_rate', type=str, choices=['True', 'False'])

parser.add_argument('--cyclic_learning_rate_min', type=float)
parser.add_argument('--cyclic_learning_rate_max', type=float)
parser.add_argument('--cyclic_learning_rate_step', type=int)

parser.add_argument('--augment', type=str, choices=['True', 'False'])

parser.add_argument('--translation', type=float)
parser.add_argument('--scale', type=float)
parser.add_argument('--rotation', type=float)
parser.add_argument('--brightness', type=float)

parser.add_argument('--model_checkpoint_monitor', type=str)
parser.add_argument('--model_checkpoint_verbose', type=int, choices=[0, 1, 2])
parser.add_argument('--model_checkpoint_mode', type=str, choices=['min', 'max'])

parser.add_argument('--model_summary', type=str, choices=['True', 'False'])

params = parser.parse_args()

params.dropout = eval(params.dropout)
params.cyclic_learning_rate = eval(params.cyclic_learning_rate)
params.augment = eval(params.augment)
params.model_summary = eval(params.model_summary)


def main():

    training_generator, validation_generator = generators.create(params)

    model = network.build(params)

    with mlflow.start_run():

        mlflow.set_tag('mlflow.runName', params.run_name)

        try:

            model.fit(
                training_generator,
                steps_per_epoch=params.training_step_size,
                epochs=params.epochs,
                verbose=1,
                callbacks=callbacks.get(params),
                validation_data=validation_generator,
                validation_steps=params.validation_step_size,
            )

        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
