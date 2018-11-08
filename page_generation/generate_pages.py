# Test for list structures in PyLaTeX.
# More info @ http://en.wikibooks.org/wiki/LaTeX/List_Structures
from pylatex import Document, Section, Itemize, Enumerate, Description, \
    Command, NoEscape
import os
import glob

import numpy as np
import cv2 as cv
from pdf2image import convert_from_path

from generic_producers import StandardUniform
from document_producer import DocumentProducer
from content_producers import EquationProducer, SentenceProducer, \
    SectionProducer, CollectionProducer, ParagraphProducer

from constants import BOX_COLOR_RGB, BOX_TEXT_COLOR_RGB, \
    OLD_TEXT_COLOR_RGB, NEW_TEXT_COLOR_RGB, \
    OLD_PAGE_COLOR_RGB, NEW_PAGE_COLOR_RGB


def open_rgb_image(path):
    img = cv.imread(path)
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)

def one_hot_image(img, threshold):
    """
    Creates new image where pixels take on the following values
    0 : pixel was below threshold
    1 : pixel was at or above threshold
    """
    ret = np.zeros(img.shape, dtype=img.dtype)
    ret[img >= threshold] = 1
    return ret

def save_rgb_image(img, path):
    cv.imwrite(path, cv.cvtColor(img, cv.COLOR_RGB2BGR))

def replace_pixels(img, from_rgb, to_rgb):
    assert len(img.shape) == 3 and img.shape[-1] == 3

    from_r, from_g, from_b = from_rgb
    img_r, img_g, img_b = img[..., 0], img[..., 1], img[..., 2]
    mask = (img_r == from_r) & (img_g == from_g) & (img_b == from_b)

    img[mask] = to_rgb

def save_page_image(page, path):
    prefix, extension = os.path.splitext(path)
    page.save(path, extension[1:].upper())

    # Image is either 0 or 255 everywhere
    img = one_hot_image(open_rgb_image(path), 128) * 255

    # Replace the pixels (text is now NEW_TEXT_COLOR, page is now NEW_PAGE_COLOR)
    replace_pixels(img, OLD_TEXT_COLOR_RGB, NEW_TEXT_COLOR_RGB)
    replace_pixels(img, OLD_PAGE_COLOR_RGB, NEW_PAGE_COLOR_RGB)

    save_rgb_image(img, path)

def save_document(doc, prefix_path, dpi=400, keep_pdf = True, keep_tex = True):

    # Generate PDF for the document
    try:
        doc.generate_pdf(prefix_path, clean_tex=(not keep_tex))
        pdf_path = f"{prefix_path}.pdf"

        # Convert the PDF into page PNGs, save them
        pages = convert_from_path(pdf_path, dpi=dpi)
        for num, page in enumerate(pages):
            png_path = f"{prefix_path}_{num}.png"
            save_page_image(page, png_path)

        # Remove the pdf if asked
        if not keep_pdf:
            os.remove(pdf_path)

    # PDF creation unsuccessful, delete anything with prefix path
    except Exception as e:
        print(e)
        for path in glob.iglob(f"{prefix_path}*"):
            os.remove(path)

def create_document(num_sections):
    doc = DocumentProducer()

    for i in range(num_sections):

        with doc.create(SectionProducer()):

            if StandardUniform() < 0.3:
                with doc.create(CollectionProducer()):
                    pass

            else:
                doc.append(ParagraphProducer())

            if StandardUniform() < 0.1:
                doc.append(EquationProducer())

        if StandardUniform() < 0.6:
            for j in range(int(abs(StandardUniform() * 5))):
                doc.append(ParagraphProducer())

            if StandardUniform() < 0.2:
                doc.append(EquationProducer())

    return doc

if __name__ == '__main__':

    for i in range(10):
        doc = create_document(5)
        save_document(doc, f'pages/{i}', keep_tex=False, keep_pdf=False)