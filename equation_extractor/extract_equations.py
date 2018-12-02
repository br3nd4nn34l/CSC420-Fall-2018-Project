import keras
import numpy as np
import cv2 as cv

from helpers.page_processing import judge_page

EROSION_AMT = 5

def refine_regions(region_mask):
    # Don't change the original array (for safety)
    ret = np.copy(region_mask)

    # Threshold, 90th percentile seems to work well
    thresh = np.percentile(ret, q=90)
    ret[ret < thresh] = 0

    # Erode since there is a bit of "fuzz" around each region
    # Also helps to get rid of small "islands"
    return cv.erode(ret, kernel=np.ones((EROSION_AMT, EROSION_AMT)))


def get_equation_bounds(region_mask):
    # Create binary version of input array
    # suitable for contour finding
    binary_arr = np.copy(region_mask)
    binary_arr[binary_arr > 0] = 255
    binary_arr = binary_arr.astype(np.uint8)[..., np.newaxis]

    # Find contours, then find bounding rectangles
    (_, contours, _) = cv.findContours(binary_arr,
                                       mode=cv.RETR_LIST,
                                       method=cv.CHAIN_APPROX_NONE)
    rectangles = (
        cv.boundingRect(cont)
        for cont in contours
    )

    # Convert rectangles to (x_min, y_min, x_max, y_max) format
    # Compensate for erosion by making rectangle a few pixels bigger
    comp = int(EROSION_AMT / 2)
    denormed = np.array([
        (x, y, x + w, y + h)
        for (x, y, w, h) in rectangles
    ]) + np.array([-comp, -comp, comp, comp])

    # Abort if the array is empty
    if denormed.size == 0:
        return []

    # 0-1 normalize the above
    h, w = region_mask.shape[:2]
    return denormed / np.array([w, h, w, h])


def extract_equations(page_img, char_judge, region_proposer):
    # Score each character's bounding box
    judgment = judge_page(
        cv.cvtColor(page_img, cv.COLOR_BGR2GRAY),
        char_judge,
        out_h=256, out_w=256
    )

    # Use region proposal model to create
    # probability image of equation-ness
    region_mask = region_proposer.predict(
        judgment[np.newaxis, ..., np.newaxis]
    )[0]

    # Refine the above mask
    refined_mask = refine_regions(region_mask)

    # Determine bounds of the equations (0-1 normalized)
    img_h, img_w = page_img.shape[:2]
    raw_bounds = get_equation_bounds(refined_mask)

    if len(raw_bounds) == 0:
        return []

    # Renormalize
    bounds = (raw_bounds *
              np.array([img_w, img_h, img_w, img_h])).astype(int)

    # Go through each rectangle in the
    # refined mask, find the sub-image
    return [
        page_img[y_min:y_max, x_min:x_max]
        for (x_min, y_min, x_max, y_max) in bounds
    ]

if __name__ == '__main__':

    # Example usage
    from helpers.data_provision import provide_page

    def load_models(judge_path, unet_path):
        return [
            keras.models.load_model(p)
            for p in [judge_path, unet_path]
        ]

    judge, unet = load_models("../models/char_judge.h5", "../models/unet.h5")
    page_dir = "../data/pages"
    name = "1064_2"
    page = provide_page(page_dir, name)

    for (i, equation) in enumerate(extract_equations(page, judge, unet)):
        cv.imwrite(f"{name}_{i}.png", equation)