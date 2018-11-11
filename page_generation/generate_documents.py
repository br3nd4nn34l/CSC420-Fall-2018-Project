# Test for list structures in PyLaTeX.
# More info @ http://en.wikibooks.org/wiki/LaTeX/List_Structures
import os

from page_generation.generic_producers import StandardUniform
from page_generation.document_producer import DocumentProducer
from page_generation.content_producers import EquationProducer, \
    SectionProducer, CollectionProducer, ParagraphProducer

from page_generation.constants import \
    NUM_DOCUMENTS, NUM_SECTIONS


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

if __name__ == '__main__':
    for i in range(NUM_DOCUMENTS):
        doc = create_document(NUM_SECTIONS)
        base_path = os.path.join("outputs", "documents", str(i))
        tex_path = document_to_tex(doc, base_path)
        tex_to_pdf(tex_path)