import os
import glob

import cv2 as cv
import numpy as np
from pdf2image import convert_from_path

from page_generation.constants import \
    OLD_TEXT_COLOR_RGB, NEW_TEXT_COLOR_RGB, \
    OLD_PAGE_COLOR_RGB, NEW_PAGE_COLOR_RGB


def one_hot_image(img, threshold):
    """
    Creates new image where pixels take on the following values
    0 : pixel was below threshold
    1 : pixel was at or above threshold
    """
    ret = np.zeros(img.shape, dtype=img.dtype)
    ret[img >= threshold] = 1
    return ret


def replace_pixels(img, from_rgb, to_rgb):
    assert len(img.shape) == 3 and img.shape[-1] == 3

    from_r, from_g, from_b = from_rgb
    img_r, img_g, img_b = img[..., 0], img[..., 1], img[..., 2]
    mask = (img_r == from_r) & (img_g == from_g) & (img_b == from_b)

    img[mask] = to_rgb


def process_page_image(page_img):
    """
    Converts page_img into an image such that:
    RED color channel is writing
    BLUE color channel is equation boxes
    """

    # Image is either 0 or 255 everywhere
    img = one_hot_image(page_img, 128) * 255

    # Replace the pixels (text is now NEW_TEXT_COLOR, page is now NEW_PAGE_COLOR)
    replace_pixels(img, OLD_TEXT_COLOR_RGB, NEW_TEXT_COLOR_RGB)
    replace_pixels(img, OLD_PAGE_COLOR_RGB, NEW_PAGE_COLOR_RGB)

    return img


def pdf_to_numpy_pages(pdf_path, dpi=300):
    """
    Converts the PDF at pdf_path into an iterator of RGB numpy arrays,
    one for each page image
    """
    pages = convert_from_path(pdf_path, dpi=dpi, thread_count=4)
    for page in pages:
        yield process_page_image(np.array(page))


def get_rectangle_coords(rect_img):
    """
    Yields the 0-1 normalized coordinates of the rectangles in rect_img.
    Each coordinate is of form:
    (x_min,y_min,x_max,y_max)
    """

    img_h, img_w = rect_img.shape

    (_, contours, _) = cv.findContours(np.copy(rect_img), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)


    for cont in contours:
        (x, y, w, h) = cv.boundingRect(cont)
        x_min, y_min, x_max, y_max = x, y, x + w, y + h

        x_min /= img_w
        y_min /= img_h
        x_max /= img_w
        y_max /= img_h

        yield (x_min, y_min, x_max, y_max)


def save_page_data(page_img, writing_path, label_path):
    # Get the relevant color channels
    writing = page_img[..., 0]  # RED CHANNEL
    rectangles = page_img[..., 2]  # BLUE CHANNEL

    # Write each rectangle coord to the rectangle file
    rect_coords = list(get_rectangle_coords(rectangles))
    with open(label_path, "w") as rect_file:
        for coord in rect_coords:
            assert len(coord) == 4
            rect_file.write(",".join(str(x) for x in coord) + "\n")

    # Invert colors of writing (so that writing is now black)
    black_writing = np.bitwise_not(writing)
    cv.imwrite(writing_path, black_writing)


if __name__ == '__main__':
    pdf_paths = glob.glob("outputs/documents/*.pdf")

    for pdf in pdf_paths:
        pdf_name = os.path.basename(pdf)
        filename = os.path.splitext(pdf_name)[0]

        for num, page in enumerate(pdf_to_numpy_pages(pdf)):
            writing_path = f"outputs/pages/{filename}_{num}.png"
            label_path = f"outputs/labels/{filename}_{num}.txt"
            save_page_data(page, writing_path, label_path)
