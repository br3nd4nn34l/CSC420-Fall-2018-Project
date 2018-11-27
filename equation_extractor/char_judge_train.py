import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))  # So this can be run as a script

import random
import argparse

random.seed(123)


import numpy as np

from helpers.data_provision import \
    provide_char_and_label, provide_character_names

from helpers.generic import chunks


def char_judge_data_generator(char_dir, char_names, batch_size):
    shuff_names = char_names[:]
    random.shuffle(shuff_names)

    while True:
        for batch in chunks(shuff_names, batch_size):
            chars_and_labels = [
                provide_char_and_label(char_dir, name)
                for name in batch
            ]

            chars, labels = zip(*chars_and_labels)

            xs = np.array(chars)
            ys = np.array(labels)

            yield (xs, ys)


def make_model(input_shape):
    import keras

    # Heavily based on code from here:
    # https://medium.com/tensorflow/hello-deep-learning-fashion-mnist-with-keras-50fcff8cd74a
    model = keras.Sequential()

    # Conv Layer 1
    model.add(keras.layers.Conv2D(
        input_shape=input_shape,
        filters=32,
        kernel_size=3,
        activation="relu",
    ))
    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(keras.layers.Dropout(0.3))

    # Conv Layer 2
    model.add(keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        activation="relu"
    ))
    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(keras.layers.Dropout(0.3))

    # Fully Connected 1
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation="relu"))
    model.add(keras.layers.Dropout(0.3))

    # Output (probability)
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    # Compile model to output probabilities
    # (output = probability given character is from an equation)
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model


def main(char_dir, model_path, epochs, batch_size):
    char_names = provide_character_names(char_dir)
    random.shuffle(char_names)

    train_prop = 0.7
    split_ind = int(len(char_names) * train_prop)
    train_names, valid_names = char_names[:split_ind], char_names[split_ind:]

    train_steps, valid_steps = [
        int(len(names) / batch_size)
        for names in [train_names, valid_names]
    ]
    train_gen, valid_gen = [
        char_judge_data_generator(char_dir, names, batch_size)
        for names in [train_names, valid_names]
    ]

    model = make_model((32, 32, 1))
    model.fit_generator(

        # Specify training generator, and how many
        # steps it should work per epoch
        generator=train_gen,
        steps_per_epoch=train_steps,

        # Specify validation generator, and how many
        # steps it should work per epoch
        validation_data=valid_gen,
        validation_steps=valid_steps,

        # Run for this many epochs
        epochs=epochs,

        shuffle=True
    )

    model.save(model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Trains a CNN classifier to score characters "
                    "based on their \"equation-ness\""
    )

    parser.add_argument(
        "char_dir",
        type=str,
        help="Directory to where the images of characters are stored."
    )

    parser.add_argument(
        "model_path",
        type=str,
        help="Where to store the trained model."
    )

    parser.add_argument(
        "epochs",
        type=int,
        help="How many epochs to train the model for."
    )

    parser.add_argument(
        "batch_size",
        type=int,
        help="Batch size for training."
    )

    args = parser.parse_args()

    main(
        char_dir=args.char_dir,
        model_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size
    )