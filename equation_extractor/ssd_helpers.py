# Used to interact with the ssd_keras library

# Adapted Encoder, Decoder, Model Code from:
# https://github.com/pierluigiferrari/ssd_keras/blob/master/ssd7_training.ipynb
# https://github.com/pierluigiferrari/ssd_keras/blob/master/ssd300_training.ipynb
import os
import keras
import numpy as np
import random

from helpers.generic import chunks, tf_model_input_sizes

random.seed(420)

from helpers.data_provision import \
    process_page, provide_page, provide_equation_labels

from ssd_keras.keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from ssd_keras.keras_layers.keras_layer_L2Normalization import L2Normalization
from ssd_keras.keras_loss_function.keras_ssd_loss import SSDLoss
from ssd_keras.ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_keras.ssd_encoder_decoder.ssd_output_decoder import decode_detections
from ssd_keras.models import keras_ssd7, keras_ssd300

# region Default Settings (from Notebook)

# Transform the input pixel values to the interval `[-1,1]`.
intensity_mean = 127.5
intensity_range = 127.5

aspect_ratios = [0.5, 1.0, 2.0]  # The list of aspect ratios for the anchor boxes
two_boxes_for_ar1 = True  # Whether or not you want to generate two anchor boxes for aspect ratio 1
steps = None  # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = None  # In case you'd like to set the offsets for the anchor box grids manually; not recommended
clip_boxes = True  # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0]  # The list of variances by which the encoded target coordinates are scaled
l2_regularization = 0.0005  # Penalize model for extreme weightings
norm_coords = True  # Model should output normalized coords

# endregion

# region Modified Settings

# Only have equations
n_classes = 1

# Internal representation of labels in the model
coords = "corners"

# Scales for bounding boxes
min_scale = 0.03
max_scale = 1.0


# endregion

def make_ssd7_model(size):
    return keras_ssd7.build_model(
        image_size=(size, size, 1),
        coords=coords,
        n_classes=n_classes,
        mode='training',
        l2_regularization=l2_regularization,
        min_scale=min_scale,
        max_scale=max_scale,
        aspect_ratios_global=aspect_ratios,
        aspect_ratios_per_layer=None,
        two_boxes_for_ar1=two_boxes_for_ar1,
        steps=steps,
        offsets=offsets,
        clip_boxes=clip_boxes,
        variances=variances,
        normalize_coords=norm_coords,
        subtract_mean=intensity_mean,
        divide_by_stddev=intensity_range
    )


def make_ssd300_model(size):
    return keras_ssd300.ssd_300(
        image_size=(size, size, 1),
        coords=coords,
        n_classes=n_classes,
        mode='training',
        l2_regularization=l2_regularization,
        min_scale=min_scale,
        max_scale=max_scale,
        aspect_ratios_global=aspect_ratios,
        aspect_ratios_per_layer=None,
        two_boxes_for_ar1=two_boxes_for_ar1,
        steps=steps,
        offsets=offsets,
        clip_boxes=clip_boxes,
        variances=variances,
        normalize_coords=norm_coords,
        subtract_mean=intensity_mean,
        divide_by_stddev=intensity_range,
        swap_channels=False
    )


def load_ssd7_model(model_path):
    """
    Attempts to load the compiled model from model_path.
    Adapted from https://github.com/pierluigiferrari/ssd_keras/blob/master/ssd7_training.ipynb
    :param model_path: path to the model
    :return: the model serialized at model_path for training
    """

    adam = keras.optimizers.Adam(
        lr=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        decay=0.0
    )

    ssd_loss = SSDLoss()

    base, ext = os.path.splitext(model_path)

    if ext.lower() == ".h5":
        model = keras.models.load_model(
            model_path,
            custom_objects={
                'compute_loss': ssd_loss.compute_loss,
                'AnchorBoxes': AnchorBoxes
            }
        )

    elif ext.lower() == ".json":
        model = keras.models.model_from_json(
            open(model_path).read(),
            custom_objects={
                "AnchorBoxes": AnchorBoxes,
            }
        )
        model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

    else:
        raise Exception("Invalid file extension. Expecting h5 or json.")

    return model


def load_ssd300_model(model_path):
    """
    Attempts to load the compiled model from model_path.
    Adapted from https://github.com/pierluigiferrari/ssd_keras/blob/master/ssd300_training.ipynb
    :param model_path: path to the model
    :return: the model serialized at model_path for training
    """

    adam = keras.optimizers.Adam(
        lr=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        decay=0.0
    )

    ssd_loss = SSDLoss()

    base, ext = os.path.splitext(model_path)

    if ext.lower() == ".h5":
        model = keras.models.load_model(
            model_path,
            custom_objects={
                'compute_loss': ssd_loss.compute_loss,
                'AnchorBoxes': AnchorBoxes,
                'L2Normalization': L2Normalization
            }
        )

    elif ext.lower() == ".json":
        model = keras.models.model_from_json(
            open(model_path).read(),
            custom_objects={
                "AnchorBoxes": AnchorBoxes,
                'L2Normalization': L2Normalization
            }
        )
        model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

    else:
        raise Exception("Invalid file extension. Expecting h5 or json.")

    return model


def ssd7_input_encoder(model):
    """
    Creates the encoder necessary to convert the
    label format of this project into the label
    format for model
    """

    # Get input dimensions
    img_height, img_width, img_channels = tf_model_input_sizes(model)[0]

    # Get predictor sizes
    predictor_sizes = [model.get_layer('classes4').output_shape[1:3],
                       model.get_layer('classes5').output_shape[1:3],
                       model.get_layer('classes6').output_shape[1:3],
                       model.get_layer('classes7').output_shape[1:3]]

    # Create encoder
    return SSDInputEncoder(
        img_height=img_height, img_width=img_width,
        coords=coords,
        n_classes=n_classes,
        predictor_sizes=predictor_sizes,
        min_scale=min_scale,
        max_scale=max_scale,
        aspect_ratios_global=aspect_ratios,
        two_boxes_for_ar1=two_boxes_for_ar1,
        steps=steps,
        offsets=offsets,
        clip_boxes=clip_boxes,
        variances=variances,
        matching_type='multi',
        pos_iou_threshold=0.5,
        neg_iou_limit=0.3,
        normalize_coords=norm_coords
    )


def ssd300_input_encoder(model):
    # Get input dimensions
    img_height, img_width, img_channels = tf_model_input_sizes(model)[0]

    # Get predictor sizes
    predictor_sizes = [
        model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
        model.get_layer('fc7_mbox_conf').output_shape[1:3],
        model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
        model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
        model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
        model.get_layer('conv9_2_mbox_conf').output_shape[1:3]
    ]

    # Create encoder
    return SSDInputEncoder(
        img_height=img_height, img_width=img_width,
        coords=coords,
        n_classes=n_classes,
        predictor_sizes=predictor_sizes,
        min_scale=min_scale,
        max_scale=max_scale,
        aspect_ratios_global=aspect_ratios,
        two_boxes_for_ar1=two_boxes_for_ar1,
        steps=steps,
        offsets=offsets,
        clip_boxes=clip_boxes,
        variances=variances,
        matching_type='multi',
        pos_iou_threshold=0.5,
        neg_iou_limit=0.5,
        normalize_coords=norm_coords
    )


def ssd_predictions(model, x, confidence_thresh=0.3,
                    iou_threshold=0.45, top_k=200):
    """
    Return prediction matrix of the model for input x.
    Each row has form:
    (confidence, x_min, y_min, x_max, y_max)
    """

    # Need to run the model's prediction through corresponding decoder
    dets = decode_detections(
        model.predict(np.array([x])),
        confidence_thresh=confidence_thresh,
        iou_threshold=iou_threshold,
        top_k=top_k,
        normalize_coords=False,  # norm_coords,
        input_coords=coords,
    )[0]

    if dets.size == 0:
        return dets

    # Skip first column as it is a class number
    return dets[:, 1:]


def encode_eqn_labels(ssd_encoder, eqn_labels):
    """
    Encodes eqn_labels of form (x_min, y_min, x_max, y_max)
    using ssd_encoder into an SSD-compatible format
    """

    num_labels = eqn_labels.shape[0]

    # Special case, run algorithm on empty matrix
    if num_labels < 1:
        return ssd_encoder([np.array([])])

    # Encoder is expecting ABSOLUTE INPUT COORDINATES
    (x_min, y_min, x_max, y_max) = (0, 1, 2, 3)
    labels = np.copy(eqn_labels)
    labels[:, (x_min, x_max)] *= ssd_encoder.img_width
    labels[:, (y_min, y_max)] *= ssd_encoder.img_height

    # Attach column of ones to front of shape
    # (SSD encoder expects class label as first element, zero is for bkg class)
    zero_col = np.ones((num_labels, 1), dtype=labels.dtype)
    ssd_labels = np.hstack((zero_col, labels))

    # Encoder is expecting list of multiple labels
    return ssd_encoder([ssd_labels])


def ssd_data_generator(model, ssd_encoder, page_dir, label_dir, page_names, batch_size):
    side_len = tf_model_input_sizes(model)[0][0]

    shuff_names = page_names[:]
    random.shuffle(shuff_names)

    while True:
        for batch in chunks(shuff_names, batch_size):
            xs = np.array([
                process_page(provide_page(page_dir, name), side_len)
                for name in batch
            ])

            eqn_labels = (
                provide_equation_labels(label_dir, name)
                for name in batch
            )

            ys = np.concatenate([
                encode_eqn_labels(ssd_encoder, label)
                for label in eqn_labels
            ], axis=0)

            yield (xs, ys)
