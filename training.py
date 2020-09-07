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

parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_norm', action='store_false')

parser.add_argument('--learning_rate', type=float, default=0.01)

parser.add_argument('--cyclic_learning_rate', action='store_false')

parser.add_argument('--cyclic_learning_rate_min', type=float, default=0.00001)
parser.add_argument('--cyclic_learning_rate_max', type=float, default=0.01)
parser.add_argument('--cyclic_learning_rate_step', type=int, default=10)

parser.add_argument('--augment', action='store_false')

parser.add_argument('--translation', type=float, default=0.2)
parser.add_argument('--scale', type=float, default=0.2)
parser.add_argument('--rotation', type=float, default=30.0)
parser.add_argument('--brightness', type=float, default=0.2)

parser.add_argument('--model_summary', action='store_true')

params = parser.parse_args()


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
