# Test for list structures in PyLaTeX.
# More info @ http://en.wikibooks.org/wiki/LaTeX/List_Structures
from pylatex import Document, Section, Itemize, Enumerate, Description, \
    Command, NoEscape

from document_producer import DocumentProducer
from latex_commands import BoxedEquation

from pdf2image import convert_from_path

import os

def save_document(doc, prefix_path, dpi=400, keep_pdf = True, keep_tex = True):

    # Generate PDF for the document
    doc.generate_pdf(prefix_path, clean_tex=(not keep_tex))
    pdf_path = f"{prefix_path}.pdf"


    # Convert the PDF into page PNGs, save them
    pages = convert_from_path(pdf_path, dpi=dpi)
    for num, page in enumerate(pages):

        png_path = f"{prefix_path}_{num}.PNG"

        # Save as PNG
        page.save(png_path, "PNG")

    # Remove the pdf if asked
    if not keep_pdf:
        os.remove(pdf_path)

if __name__ == '__main__':
    for i in range(10):
        doc = DocumentProducer()

        doc.append(BoxedEquation(r"$\pi r^2$"))

        # create a bulleted "itemize" list like the below:
        # \begin{itemize}
        #   \item The first item
        #   \item The second item
        #   \item The third etc \ldots
        # \end{itemize}

        with doc.create(Section('"Itemize" list')):
            with doc.create(Itemize()) as itemize:
                itemize.add_item("the first item")
                itemize.add_item("the second item")
                itemize.add_item("the third etc")
                itemize.add_item(BoxedEquation(r"$\psi \pi \alpha$"))
                # you can append to existing items
                itemize.append(Command("ldots"))

        # create a numbered "enumerate" list like the below:
        # \begin{enumerate}[label=\alph*),start=20]
        #   \item The first item
        #   \item The second item
        #   \item The third etc \ldots
        # \end{enumerate}

        with doc.create(Section('"Enumerate" list')):
            with doc.create(Enumerate(enumeration_symbol=r"\alph*)",
                                      options={'start': 20})) as enum:
                enum.add_item("the first item")
                enum.add_item("the second item")
                enum.add_item(NoEscape("the third etc \\ldots"))

        # create a labelled "description" list like the below:
        # \begin{description}
        #   \item[First] The first item
        #   \item[Second] The second item
        #   \item[Third] The third etc \ldots
        # \end{description}

        with doc.create(Section('"Description" list')):
            with doc.create(Description()) as desc:
                desc.add_item("First", "The first item")
                desc.add_item("Second", "The second item")
                desc.add_item("Third", NoEscape("The third etc \\ldots"))

        save_document(doc, f'pages/doc_{i}', keep_tex=False, keep_pdf=False)