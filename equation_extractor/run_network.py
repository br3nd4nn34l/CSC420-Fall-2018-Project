import os
import cv2 as cv
import numpy as np

from equation_extractor.data_provision import \
    process_page, provide_page, provide_page_names, provide_labels

from equation_extractor.ssd_helpers import \
    load_ssd_model, ssd_predictions, get_model_input_size

data_path = os.path.join("..", "page_generation", "outputs")

names = provide_page_names(data_path)


model = load_ssd_model("ssd7_trained_3.h5", 1)
square_size = get_model_input_size(model)[0]


def draw_rectangles(img, rel_coords, color, thickness):

    height, width = img.shape[:2]
    abs_coords = rel_coords * np.array([width, height, width, height])

    for x_min, y_min, x_max, y_max in abs_coords:
        dn_lft = (int(x_min), int(y_min))
        up_rgt = (int(x_max), int(y_max))

        cv.rectangle(img, dn_lft, up_rgt, color, thickness)


for name in names:
    orig_page = provide_page(data_path, name)
    preds = ssd_predictions(model, process_page(orig_page, square_size))

    if preds.size > 0:
        print(name)

        preds = preds[:, 1:]
        labels = provide_labels(data_path, name)

        print("Truth:")
        print(labels)

        print("Prediction:")
        print(preds)
        print("")

        rect_img = np.copy(orig_page)
        draw_rectangles(rect_img, labels, (255, 0, 0), 2)
        draw_rectangles(rect_img, preds, (0, 0, 255), 2)

        cv.imwrite(f"{name}_rect.png", rect_img)