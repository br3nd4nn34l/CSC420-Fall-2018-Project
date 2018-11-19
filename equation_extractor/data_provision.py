import glob
import os

import cv2 as cv
import numpy as np


# Functions to present data for network training / network configuration

def process_page(img, square_size):
    return cv.resize(
        img,
        (square_size, square_size),
        interpolation=cv.INTER_AREA
    )[:, :, :1]


def provide_page_names(data_path):
    """
    Gets the name (excluding extension) of images that match:
    {data_path}/pages/*.png
    """
    page_paths = glob.glob(os.path.join(data_path, "pages", "*.png"))

    name_plus_ext = (
        os.path.basename(x)
        for x in page_paths
    )

    return [
        os.path.splitext(x)[0]
        for x in name_plus_ext
    ]


def provide_page(data_path, page_name):
    """
    Provides a numpy image of the page stored at
    page_generation/outputs/pages/{page_name}.png
    """
    page_path = os.path.join(
        data_path,
        "pages", "{}.png".format(page_name)
    )

    return cv.imread(page_path)


def provide_labels(data_path, page_name):
    """
    Provides a numpy array of bounding box labels of size (n x 4),
    where each row is of form (x_min, y_min, x_max, y_max)

    These are 0-1 normalized to the width/height of the source image.
    """
    label_path = os.path.join(
        data_path,
        "labels", "{}.txt".format(page_name)
    )

    with open(label_path, mode="r") as file:
        return np.array([
            [float(x) for x in l.split(",")]
            for l in file.readlines()
        ])
