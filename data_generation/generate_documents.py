# More info @ http://en.wikibooks.org/wiki/LaTeX/List_Structures
import os
import sys

# So this can be run as a script
sys.path.append(os.path.dirname(sys.path[0]))

import argparse

from data_generation.generic_producers import StandardUniform
from data_generation.document_producer import DocumentProducer
from data_generation.content_producers import EquationProducer, \
    SectionProducer, CollectionProducer, ParagraphProducer


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


def document_to_tex(doc, base_path):
    """
    Converts the PyLatex Document doc into a TEX file (base_path.tex).
    Returns the destination TEX path.
    """
    doc.generate_tex(base_path)

    tex_path = f"{base_path}.tex"
    if os.path.isfile(tex_path):
        return tex_path

    return None


def tex_to_pdf(tex_path):
    """
    Converts the TEX document at tex_path into a LATEX PDF (may not fully compile).
    Returns the destination PDF path.
    """
    base_path, tex = os.path.splitext(tex_path)
    try:
        os.system(f"latexmk {tex_path} -pdf -xelatex -interaction=nonstopmode -f -quiet -jobname={base_path} -p-")
    except:
        pass

    # Remove leftover stuff
    for ext in ["tex", "xdv", "aux", "fdb_latexmk", "fls", "log"]:
        to_delete = f"{base_path}.{ext}"
        if os.path.isfile(to_delete):
            os.remove(to_delete)

    pdf_path = f"{base_path}.pdf"
    if os.path.isfile(pdf_path):
        return pdf_path

    return None


def main(num_documents, num_sections, out_dir):
    for i in range(num_documents):
        doc = create_document(num_sections)
        base_path = os.path.join(out_dir, str(i))
        tex_path = document_to_tex(doc, base_path)
        tex_to_pdf(tex_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=f"Randomly generates PDF LaTeX documents for training the equation extractor."
    )

    parser.add_argument(
        "num_documents",
        type=int,
        help="Number of documents to generate."
    )

    parser.add_argument(
        "num_sections",
        type=int,
        help="Number of sections per document to generate."
    )

    parser.add_argument(
        "out_dir",
        type=str,
        help="Directory to output the LaTeX documents"
    )

    args = parser.parse_args()

    main(
        num_documents=args.num_documents,
        num_sections=args.num_sections,
        out_dir=args.out_dir
    )
