import cv2 as cv
import numpy as np

from helpers.generic import tf_model_input_sizes


def get_letter_blob_dims(img):
    """
    Return a 2D numpy array, where each row
    represents the rectangular bounds of a blob in img.

    Rows have form:
    [x, y, w, h]
    """

    # Find all contours
    (_, contours, _) = cv.findContours(img,
                                       cv.RETR_LIST,
                                       cv.CHAIN_APPROX_TC89_L1)

    # Figure out rectangles around contours
    return np.array([
        cv.boundingRect(cont)
        for cont in contours
    ])


def eliminate_outlier_blobs(blob_dims, img_h, img_w, threshold):
    """
    Eliminate blobs from blob_dims whose total area is less than
    threshold * img_h * img_w

    Return a new array that is blob_dims with the outliers removed
    """
    widths, heights = blob_dims[:, 2], blob_dims[:, 3]
    areas = widths * heights

    return blob_dims[np.where(
        areas <= img_w * img_h * threshold
    )]


def cartesian_pairs(arr):
    """
    Return the 2-cartesian product of arr in form
    (c1, c2)
    Where:
        c1[i] = first element of arr in pair i
        c2[i] = second element of arr in pair i
    Note: arr[i] and arr[i] are never combined
    """

    inds = np.arange(arr.shape[0])

    row_inds, col_inds = np.meshgrid(inds, inds)
    return arr[row_inds.flatten()], arr[col_inds.flatten()]


def encapsulated(rect_corners):
    """
    Returns all i s.t.
    rects[i] is encapsulated some other rectangle in rects.
    """

    left_rects, right_rects = cartesian_pairs(rect_corners)

    L_xmin, L_ymin, L_xmax, L_ymax = left_rects.transpose()
    R_xmin, R_ymin, R_xmax, R_ymax = right_rects.transpose()

    # Figure out where X, Y are encapsulated
    x_encapsulates = np.logical_and(L_xmin <= R_xmin, R_xmax <= L_xmax)
    y_encapsulates = np.logical_and(L_ymin <= R_ymin, R_ymax <= L_ymax)

    # Where both X, Y are encapsulated (in cartesian's indices)
    flat_inds = np.nonzero(np.logical_and(
        x_encapsulates,
        y_encapsulates
    ))

    # Above translated back to pairwise indices
    large_rects, small_rects = np.vstack(np.unravel_index(
        flat_inds,
        (rect_corners.shape[0], rect_corners.shape[0])
    ))

    # Return the indices where the rectangles are different
    return large_rects[large_rects != small_rects]


def eliminate_encapsulated_blobs(blob_corners):
    """
    Eliminates blobs from blob_dims that are
    encapsulated by other blobs.

    Returns a 2D array of blob corners,
    each element is of form:
    [x_min, y_min, x_max, y_max]

    """

    # Delete encapsulated blobs
    return np.delete(
        blob_corners,
        encapsulated(blob_corners),
        axis=0
    )


def convert_dims_to_corners(blob_dims):
    """
    Converts array whose elements are of
    form [x, y, w, h] into array whose elements
    are of form [x_min, y_min, x_max, y_max]
    """
    widths, heights = blob_dims[:, 2], blob_dims[:, 3]
    x_min, y_min = blob_dims[:, 0], blob_dims[:, 1]
    x_max, y_max = x_min + widths, y_min + heights
    return np.array([x_min, y_min, x_max, y_max]).transpose()


def get_letter_blob_corners(blob_dims, img_h, img_w):
    """
    Return a 2D numpy array where each element is of form:
    [x_min, y_min, x_max, y_max]
    """

    uniq_blob_dims = np.unique(blob_dims, axis=0)
    small_blob_dims = eliminate_outlier_blobs(
        uniq_blob_dims, img_h, img_w, threshold=0.2
    )
    blob_corners = convert_dims_to_corners(small_blob_dims)

    return eliminate_encapsulated_blobs(blob_corners)


def refine_letter_blob(black_letter, square_size):
    # Gathering shape info
    height, width = black_letter.shape

    # Assuming black letter on white background
    inverted = np.bitwise_not(black_letter)

    # Abort if the image is just black
    if np.all(inverted == 0):
        return None

    # Calculate center of mass of non-zero coordinates (the letter)
    mass_center = np.vstack(np.nonzero(inverted)) \
        .mean(axis=1)
    cent_row, cent_col = np.around(mass_center).astype(int)

    sq_side = max(height + cent_row, width + cent_col)
    ret = np.zeros((sq_side, sq_side), dtype=black_letter.dtype)

    try:

        # Figure out which index to insert the letter
        # at such that it is centered WRT
        # the center of gravity
        ret_cent_row = ret_cent_col = int(sq_side / 2)
        row_diff = ret_cent_row - cent_row
        col_diff = ret_cent_col - cent_col
        start_row, start_col = row_diff, col_diff
        end_row = start_row + height
        end_col = start_col + width
        ret[start_row:end_row, start_col:end_col] = inverted
    except:
        ret = inverted

    return cv.resize(ret, (square_size, square_size))


def extract_corners_and_letters(black_text_img, letter_size):
    """
    :param black_text_img: image of black text
    on a white background

    :param letter_size: how large to make
    each extracted letter

    Yield 2-tuples of form:

        corners: 1D numpy array of the letter's corners
        (pixel coordinates in original page)
        (x_min, y_min, x_max, y_max)

        letter: 2D numpy array of letter_size x letter_size of the
        a color-inverted letters (white on black)
    """
    blob_dims = get_letter_blob_dims(black_text_img)
    letter_corners = get_letter_blob_corners(blob_dims, *black_text_img.shape[:2])

    for corners in letter_corners:

        # Extract the letter and refine it
        (x_min, y_min, x_max, y_max) = corners
        letter = black_text_img[y_min:y_max, x_min:x_max]
        refined_letter = refine_letter_blob(letter, letter_size)

        # None means it was just a monochrome patch, skip it
        if refined_letter is None:
            continue

        # Yield the corners and refined letter
        yield corners, refined_letter


def judge_page(page, judge_model, out_h, out_w):

    # Assume the model takes in a square
    judge_inp_size = tf_model_input_sizes(judge_model)[0][0]

    # Extract corners and letters
    # according to input size
    corners, letters = [
        np.array(x)
        for x in zip(*list(extract_corners_and_letters(page, judge_inp_size)))
    ]

    # Score the letters with the judge
    letter_scores = judge_model.predict(
        letters.reshape(*letters.shape, 1)
    ).flatten()

    # Make new image
    ret = np.zeros_like(page)
    for ((x_min, y_min, x_max, y_max), score) in zip(corners, letter_scores):
        ret[y_min:y_max, x_min:x_max] = 255 * score

    # Resize for output
    # Don't do any "fuzzy" interpolation -
    # scores shouldn't be blurred together!
    return cv.resize(ret, (out_h, out_w),
                     interpolation=cv.INTER_NEAREST)