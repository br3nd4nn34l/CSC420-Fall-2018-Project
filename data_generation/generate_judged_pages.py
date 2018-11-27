import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))  # So this can be run as a script

import argparse
import cv2 as cv

import random
random.seed(123)

from helpers.page_processing import \
    judge_page

from helpers.data_provision import provide_page, provide_page_names


def load_model(model_path):
    import keras
    return keras.models.load_model(model_path)


def main(page_dir, out_dir, judge_path, num_pages):
    judge_model = load_model(judge_path)


    all_page_names = provide_page_names(page_dir)
    random.shuffle(all_page_names)
    page_names = all_page_names[:num_pages]

    for name in page_names:
        try:
            # Read the page, then judge it char-by-char
            page = cv.cvtColor(provide_page(page_dir, name), cv.COLOR_BGR2GRAY)
            judged = judge_page(page, judge_model, out_h=300, out_w=300)

            # Save the judged page
            out_path = os.path.join(out_dir, f"{name}.png")
            cv.imwrite(out_path, judged)

            print(name)
        except:
            continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=f"Judges pages in the given directory "
                    f"using the given character CNN judge. "
                    f"Judgments are stored as 300x300 images."
    )

    parser.add_argument(
        "page_dir",
        type=str,
        help="Directory that contains the PNG images of the pages."
    )

    parser.add_argument(
        "out_dir",
        type=str,
        help="Directory to output the 300x300 judged pages into."
    )

    parser.add_argument(
        "judge_path",
        type=str,
        help="Where the trained CNN judge is stored."
    )

    parser.add_argument(
        "num_pages",
        type=int,
        help="Number of pages to produce"
    )

    args = parser.parse_args()

    main(
        page_dir=args.page_dir,
        out_dir=args.out_dir,
        judge_path=args.judge_path,
        num_pages=args.num_pages
    )