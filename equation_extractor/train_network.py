import os
import sys

# Partially adapted from:
# https://github.com/pierluigiferrari/ssd_keras/blob/master/ssd7_training.ipynb

# So this can be run as a script
sys.path.append(os.path.dirname(sys.path[0]))

import argparse

from equation_extractor.data_provision import \
    provide_page_names

from equation_extractor.ssd_helpers import \
    load_ssd_model, ssd_data_generator

from keras.callbacks import ReduceLROnPlateau


def train_model(model_path, data_path, dest_path,
                epochs, epoch_steps, batch_size,
                val_prop=0.2):
    # Load the model, create the corresponding encoder
    model = load_ssd_model(model_path, batch_size)

    # Get page names, split into training and validation
    page_names = provide_page_names(data_path)
    num_data = len(page_names)
    split_ind = int((1 - val_prop) * num_data)
    train_names, val_names = page_names[:split_ind], page_names[split_ind:]

    # Create training and validation generators
    train_gen, val_gen = [
        ssd_data_generator(model, data_path, names, batch_size)
        for names in [train_names, val_names]
    ]

    # Reduce learning rate if the model hits a wall
    reduce_lr = ReduceLROnPlateau(monitor='loss',
                                  factor=0.2,
                                  patience=2, # Wait for 2 epochs
                                  verbose=1,
                                  min_delta=0.001,
                                  cooldown=0,
                                  min_lr=0.00001)

    # Run limited validation steps (proportional to percentage of training data covered)
    val_steps = (val_prop * num_data / batch_size) * \
                       ((batch_size * epoch_steps) / ((1 - val_prop) * num_data))

    # Train the model using the generator
    model.fit_generator(
        generator=train_gen,
        validation_data=val_gen,
        epochs=epochs,
        steps_per_epoch=epoch_steps,
        validation_steps=int(val_steps),
        shuffle=True,
        callbacks=[reduce_lr],
        max_queue_size=20
    )

    model.save(dest_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=f"Trains an SSD7 model to detect equation-bounding-boxes"
    )

    parser.add_argument(
        "model",
        type=str,
        help="Path to the model to train. "
             "JSON files are assumed to be architecture files - instantiate network in this case."
             "h5 files are assumed to be compiled/weighted models - resume training in this case."
    )

    parser.add_argument(
        "destination",
        type=str,
        help="Path to save the trained model at."
    )

    parser.add_argument(
        "data",
        type=str,
        help="Path to the training data directory. "
             "Must have pages and labels subdirectories, each with parallel files"
    )

    parser.add_argument(
        "epochs",
        type=int,
        help="Number of epochs to train the model for"
    )

    parser.add_argument(
        "epoch_steps",
        type=int,
        help="Number of steps (batches) in each epoch"
    )

    parser.add_argument(
        "batch_size",
        type=int,
        help="Number of samples per batch"
    )

    args = parser.parse_args()
    train_model(
        model_path=args.model,
        data_path=args.data,
        dest_path=args.destination,
        epochs=args.epochs,
        epoch_steps=args.epoch_steps,
        batch_size=args.batch_size
    )
