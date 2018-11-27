import glob
import os

import cv2 as cv
import numpy as np


# Functions to provide data

def process_page(img, square_size):
    resized = cv.resize(
        img,
        (square_size, square_size),
        interpolation=cv.INTER_AREA
    )[:, :, :1]

    # Return inverted color, gives better results
    return np.bitwise_not(resized)


def base_names(raw_paths, want_ext):
    """
    Returns a list of base names of the paths in raw_paths.

    :param raw_paths: list of paths
    :param want_ext: whether or not to include the
    extension in the base name

    :return: list of the base names of the paths in raw_paths
    """

    base_and_exts = [
        os.path.basename(p)
        for p in raw_paths
    ]

    if want_ext:
        return base_and_exts
    else:
        return [
            os.path.splitext(p)[0]
            for p in base_and_exts
        ]


def provide_document_names(doc_dir):
    """
    Gets the names (excluding extension) of documents that match:
    {doc_dir}/*.pdf
    """
    return base_names(
        glob.glob(os.path.join(doc_dir, "*.pdf")),
        want_ext=False
    )


def provide_page_names(page_dir):
    """
    Gets the names (excluding extension) of images that match:
    {page_dir}/*.png
    """
    return base_names(
        glob.glob(os.path.join(page_dir, "*.png")),
        want_ext=False
    )


def provide_page(page_dir, page_name):
    """
    Provides a numpy image of the page stored at
    {page_dir}/{page_name}.png
    """
    page_path = os.path.join(
        page_dir, "{}.png".format(page_name)
    )
    return cv.imread(page_path)


def provide_character_names(char_dir):
    return base_names(
        glob.glob(os.path.join(char_dir, "*.png")),
        want_ext=False
    )


def provide_equation_labels(eqn_label_dir, page_name):
    """
    Reads the label file {eqn_label_dir}/{page_name}/png

    Returns a numpy array of bounding box labels of size (n x 4),
    where each row is of form (x_min, y_min, x_max, y_max)

    These are 0-1 normalized to the width/height of the source image.
    """
    label_path = os.path.join(
        eqn_label_dir,
        "{}.txt".format(page_name)
    )

    with open(label_path, mode="r") as file:
        return np.array([
            [float(x) for x in l.split(",")]
            for l in file.readlines()
        ])


def provide_char_and_label(char_dir, char_name):
    _, label_str = char_name.split("_")
    img_path = os.path.join(char_dir, f"{char_name}.png")
    char_img = cv.imread(img_path)[:, :, :1]

    return char_img, int(label_str)