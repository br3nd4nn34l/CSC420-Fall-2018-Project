import os
import sys

# So this can be run as a script
sys.path.append(os.path.dirname(sys.path[0]))

# Partially adapted from:
# https://github.com/pierluigiferrari/ssd_keras/blob/master/ssd300_training.ipynb

import argparse

from helpers.data_provision import \
    provide_page_names

from equation_extractor.ssd_helpers import \
    load_ssd300_model, ssd300_input_encoder, \
    ssd_data_generator

from keras.callbacks import \
    ReduceLROnPlateau, LearningRateScheduler, \
    ModelCheckpoint


def make_lr_schedule(epoch_steps):

    def ret(epoch):
        steps = epoch * epoch_steps

        if steps < 5000:
            return 0.001
        elif steps < 20000:
            return 0.0001
        else:
            return 0.00001

    return ret


def main(model_path, page_dir, label_dir, dest_path,
         epochs, batch_size, init_epoch, val_prop=0.2):
    # Load the model, create the corresponding encoder
    model = load_ssd300_model(model_path)

    # Get page names, split into training and validation
    page_names = provide_page_names(page_dir)
    num_data = len(page_names)
    split_ind = int((1 - val_prop) * num_data)
    train_names, val_names = page_names[:split_ind], page_names[split_ind:]

    # Create training and validation generators
    ssd300_encoder = ssd300_input_encoder(model)
    train_gen, val_gen = [
        ssd_data_generator(
            model, ssd300_encoder,
            page_dir, label_dir,
            names, batch_size
        )
        for names in [train_names, val_names]
    ]



    # Determine how many steps to do
    train_steps, val_steps = [
        int(len(names) / batch_size)
        for names in [train_names, val_names]
    ]

    # Model checkpoint
    checkpt = ModelCheckpoint(
        dest_path + "_{epoch:02d}_{val_loss:.2f}.h5",
        save_best_only=True
    )

    # Schedule learning rate
    lr_sched = LearningRateScheduler(schedule=make_lr_schedule(train_steps),
                                     verbose=1)

    # Reduce learning rate if the model hits a wall
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.2,
                                  patience=2,  # Wait for 2 epochs
                                  verbose=1,
                                  min_delta=0.001,
                                  cooldown=2,  # Keep training on rate for 2 epochs
                                  min_lr=0.00001)

    # Train the model using the generator
    model.fit_generator(
        generator=train_gen,
        validation_data=val_gen,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        shuffle=True,
        callbacks=[checkpt, lr_sched, reduce_lr],
        max_queue_size=20,
        initial_epoch=init_epoch
    )

    model.save(dest_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=f"Trains an SSD model to detect equation-bounding-boxes"
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
        help="Path to save the trained H5 models at (exclude extension)."
    )

    parser.add_argument(
        "page_dir",
        type=str,
        help="Path to the training image directory. "
             "Must have file names that run parallel to label_dir."
    )

    parser.add_argument(
        "label_dir",
        type=str,
        help="Path to the training label directory. "
             "Must have file names that run parallel to page_dir."
    )

    parser.add_argument(
        "epochs",
        type=int,
        help="Number of epochs to train the model for"
    )

    parser.add_argument(
        "batch_size",
        type=int,
        help="Number of samples per batch"
    )

    parser.add_argument(
        "initial_epoch",
        type=int,
        required=False,
        default=0,
        help="Epoch to start training on (default 0)."
    )

    args = parser.parse_args()
    main(
        model_path=args.model,
        page_dir=args.page_dir,
        label_dir=args.label_dir,
        dest_path=args.destination,
        epochs=args.epochs,
        batch_size=args.batch_size,
        init_epoch=args.initial_epoch
    )
