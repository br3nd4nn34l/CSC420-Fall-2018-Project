import keras
import numpy as np
import cv2 as cv

from helpers.page_processing import judge_page
from helpers.data_provision import \
    provide_page_names, provide_page

from equation_extractor.ssd_helpers import ssd_predictions, load_ssd300_model


def extract_equations(page_img, judge_model, ssd_model):
    judgment = judge_page(
        cv.cvtColor(page_img, cv.COLOR_BGR2GRAY),
        judge_model,
        out_h=300, out_w=300
    )

    predictions = ssd_predictions(ssd_model, np.dstack((judgment,)))

    ret = np.copy(page_img)
    img_h, img_w = ret.shape[:2]
    for (conf, x_min, y_min, x_max, y_max) in predictions:

        corner1, corner2 = [
            (int(x * img_w), int(y * img_w))
            for (x, y) in [(x_min, y_min), (x_max, y_max)]
        ]
        cv.rectangle(ret, corner1, corner2, (0, 255, 0), int(conf * 4))

    return ret

def load_models(judge_path, ssd_path):
    return keras.models.load_model(judge_path), \
           load_ssd300_model(ssd_path)

judge, ssd = load_models("../models/char_judge.h5", "../models/ssd300_28_2.23.h5")
page_dir = "../data/pages"
name = "28_7"
page = provide_page(page_dir, name)

cv.imwrite(f"{name}.png", extract_equations(page, judge, ssd))