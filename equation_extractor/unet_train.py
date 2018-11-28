import os
import sys

# So this can be run as a script
sys.path.append(os.path.dirname(sys.path[0]))

import argparse
import numpy as np
import cv2 as cv
import random

from helpers.generic import chunks
from helpers.data_provision import \
    provide_page_names, provide_page, \
    provide_equation_labels


def label_mask(labels, img_h, img_w):
    ret = np.zeros((img_h, img_w), dtype=np.float32)

    if labels.size == 0:
        return ret

    denormed_labels = (labels * np.array([img_w, img_h, img_w, img_h]))\
        .astype(int)
    for (x_min, y_min, x_max, y_max) in denormed_labels:
        ret[y_min:y_max, x_min:x_max] = 1

    return ret


def unet(input_size=(256, 256, 1)):
    """
    Copied from https://github.com/zhixuhao/unet/blob/master/model.py
    :param input_size:
    :return:
    """
    # Import stuff here, otherwise Tensorflow
    # starts as soon as we boot, even for bad arguments
    from keras.models import Model
    from keras.layers import \
        Conv2D, MaxPooling2D, UpSampling2D, \
        Dropout, concatenate, Input
    from keras.optimizers import Adam

    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    return model


def unet_data_generator(page_dir, label_dir, name_lst, batch_size):
    while True:
        for names in chunks(name_lst, batch_size):

            xs = np.array([
                cv.resize(
                    provide_page(page_dir, n),
                    (256, 256),
                    interpolation=cv.INTER_NEAREST
                )[..., 0]
                for n in names
            ])[..., np.newaxis]

            labels = (
                provide_equation_labels(label_dir, n)
                for n in names
            )

            ys = np.array([
                label_mask(l, 256, 256)
                for l in labels
            ])[..., np.newaxis]

            yield (xs, ys)


def main(page_dir, label_dir, model_path, batch_size, epochs):
    names = provide_page_names(page_dir)
    random.seed(123)
    random.shuffle(names)

    train_prop = 0.8
    split_ind = int(len(names) * train_prop)
    train_names, val_names = names[:split_ind], names[split_ind:]

    # Create generators
    train_gen, val_gen = [
        unet_data_generator(page_dir, label_dir, n, batch_size)
        for n in [train_names, val_names]
    ]

    # How many steps to run
    train_steps, val_steps = [
        int(len(n) / batch_size)
        for n in [train_names, val_names]
    ]

    model = unet()

    model.fit_generator(
        generator=train_gen,
        validation_data=val_gen,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        shuffle=True,
        max_queue_size=20
    )

    model.save(model_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=f"Creates and trains a 256x256x1 input U-Net model designed to detect "
                    f"equation regions, given a character-judged page."
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
        "model_path",
        type=str,
        help="Path to save the trained model at."
    )

    parser.add_argument(
        "batch_size",
        type=int,
        help="Number of samples per batch."
    )

    parser.add_argument(
        "epochs",
        type=int,
        help="Number of epochs to train the model for."
    )

    args = parser.parse_args()
    main(
        page_dir=args.page_dir,
        label_dir=args.label_dir,
        model_path=args.model_path,
        batch_size=args.batch_size,
        epochs=args.epochs
    )