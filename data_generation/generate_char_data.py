import os
import sys

from helpers.page_processing import extract_corners_and_letters

sys.path.append(os.path.dirname(sys.path[0]))  # So this can be run as a script

import cv2 as cv
import argparse
import random

from helpers.data_provision import \
    provide_page, provide_page_names, \
    provide_equation_labels


def is_letter_in_equation(letter_corners, equation_labels):
    """
    Return whether or not letter_corners are
    inside any of the rectangles specified by
    in equation_labels.

    Assumption: all coordinates are in the same scale.

    :param letter_corners: (x,y) coordinates of the
    corners of the given letter

    :param equation_labels: (x,y) coordinates of the
     corners that determine each of the equation labels

    :return: boolean indicating whether or not
    letter_corners are inside any of the rectangles
    specified by in equation_labels.
    """

    let_x_min, let_y_min, \
    let_x_max, let_y_max = letter_corners

    for eq_x_min, eq_y_min, \
        eq_x_max, eq_y_max in equation_labels:

        x_enclosed = (eq_x_min <= let_x_min) and (let_x_max <= eq_x_max)
        y_enclosed = (eq_y_min <= let_y_min) and (let_y_min <= eq_y_max)

        if (x_enclosed and y_enclosed):
            return True

    return False


def extract_labelled_letters(black_text_img, equation_labels, letter_size):
    """
    :param black_text_img: image of black text
    on a white background

    :param equation_labels: 2D numpy array with rows of form
    [x_min, y_min, x_max, y_max] (0-1 normalized to image dimensions)

    :param letter_size: how large to make each
    extracted letter

    Yield tuples of form:
        Boolean: whether or not the letter's rectangle
        was inside an equation label

        Numpy Array: letter_size x letter_size of the
            color-inverted letters (white on black)
    """

    # Denormalize the equation labels
    img_h, img_w = black_text_img.shape[:2]
    eq_labels = equation_labels
    if eq_labels.size > 0:
        eq_labels *= [img_w, img_h, img_w, img_h]

    for (corners, letter) in extract_corners_and_letters(black_text_img, letter_size):
        in_eq = is_letter_in_equation(corners, eq_labels)
        yield in_eq, letter


def main(page_dir, eqn_label_dir, char_dir, num_chars):
    # Shuffle then restrict page names
    random.seed(123)
    page_names = provide_page_names(page_dir)
    random.shuffle(page_names)

    # Track how many of each class we have
    num_eq, num_neq = 0, 0

    for name in page_names:
        print(name)

        page = cv.cvtColor(provide_page(page_dir, name),
                           cv.COLOR_BGR2GRAY)
        eqn_labels = provide_equation_labels(eqn_label_dir, name)

        letters = extract_labelled_letters(page, eqn_labels, 32)

        for (in_eqn, letter) in letters:

            # Abort if total exceeds num_total
            if num_eq + num_neq > num_chars:
                return

            # Skip to balance out classes
            if in_eqn and (num_eq > (num_chars / 2)):
                continue
            if not in_eqn and (num_neq > (num_chars / 2)):
                continue

            # Write the file (include label data in name)
            char_path = os.path.join(char_dir,
                                     f"{num_eq + num_neq}_{int(in_eqn)}.png")
            cv.imwrite(char_path, letter)

            # Increment appropriate counters
            if in_eqn:
                num_eq += 1
            else:
                num_neq += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Converts page PNGs and equation label TXTs into "
                    "individual 32x32 letters needed to train the classifier. "
                    "Characters are named as follows: {char_num}_{in equation}"
    )

    parser.add_argument(
        "page_dir",
        type=str,
        help="Directory that contains the PNG images of pages."
    )

    parser.add_argument(
        "eqn_label_dir",
        type=str,
        help="Directory to where equation labels are stored as TXT. "
             "Names run parallel to page_dir."
    )

    parser.add_argument(
        "char_dir",
        type=str,
        help="Directory to output the images of characters into."
    )

    parser.add_argument(
        "num_chars",
        type=int,
        help="Number of characters to produce"
    )

    args = parser.parse_args()

    main(
        page_dir=args.page_dir,
        eqn_label_dir=args.eqn_label_dir,
        char_dir=args.char_dir,
        num_chars=args.num_chars
    )
