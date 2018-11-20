import cv2
import numpy as np


def fix_corners(corners):
    """
    Corners are given in any ordering. Fix them so that the ordering is:
    [top left, top right, bottom left, bottom right].
    """
    tl, tr, bl, br = None, None, None, None
    corners = corners.reshape(4, 2)
    sorted_x = sorted(corners, key=lambda x: x[0])
    left, right = sorted_x[:2], sorted_x[2:]

    if left[0][1] > left[1][1]:
        tl, bl = left[1], left[0]
    else:
        tl, bl = left[0], left[1]

    if right[0][1] > right[1][1]:
        tr, br = right[1], right[0]
    else:
        tr, br = right[0], right[1]

    return [tl, tr, bl, br]


def homography(corners, img, new_size=None):
    """
    Given the corners of a page in the image, perform a homography transformation
    to extract the page.
    """
    if not new_size:
        new_size = img.shape
    new_img = np.zeros(new_size)

    print(corners)
    corners = np.array(fix_corners(corners))

    fix_corners(corners)

    new_y, new_x, _ = new_img.shape
    new_corners = np.array([(0, 0), (new_x, 0), (0, new_y), (new_x, new_y)])

    # Calculate Homography
    h, status = cv2.findHomography(corners, new_corners)

    # Warp source image to destination based on homography
    im_out = cv2.warpPerspective(img, h, (img.shape[1], img.shape[0]))

    return im_out