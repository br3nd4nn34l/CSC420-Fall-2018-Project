from pylatex import Document, NoEscape

from data_generation.generic_producers import \
    DictProducer, ListProducer, \
    InstanceProducer, \
    UniformProducer, NormalProducer

from data_generation.constants import COLOR_MODEL, \
    BOX_COLOR_NAME, BOX_COLOR_RGB, \
    BOX_TEXT_COLOR_NAME, BOX_TEXT_COLOR_RGB

from data_generation.latex_commands import BoxedEquationDefinition

# Picks document class
DocumentClassProducer = ListProducer(["article", "report"]) \
    .then(lambda lst: lst[0])

# Picks font size
FontSizeProducer = ListProducer([
    "Huge", "huge",
    "LARGE", "Large", "large",
    "normalsize",
    "small", "footnotesize", "scriptsize", "tiny"
]) \
    .then(lambda lst: lst[0])

# Picks margins
MarginProducer = NormalProducer(20, 10) \
    .then(lambda x: str(int(abs(x))) + "mm")

# Picks page layout parameters
GeometryProducer = DictProducer(
    left=MarginProducer,
    right=MarginProducer,
    top=MarginProducer,
    bottom=MarginProducer
)


# Helper function for adding colors to doc, returns the modified document
def add_color(doc, name, rgb_tuple):
    doc.add_color(name,
                  COLOR_MODEL,
                  ", ".join(str(x) for x in rgb_tuple))
    return doc


def doc_append(doc, cmd):
    doc.append(cmd)
    return doc


# Produces LATEX document objects
DocumentProducer = InstanceProducer(
    Document,
    documentclass=DocumentClassProducer,
    font_size=FontSizeProducer,
    page_numbers=UniformProducer(0, 1)
        .then(lambda x: x > 0.5),
    geometry_options=GeometryProducer
) \
    .then(lambda doc: add_color(doc, BOX_COLOR_NAME, BOX_COLOR_RGB)) \
    .then(lambda doc: add_color(doc, BOX_TEXT_COLOR_NAME, BOX_TEXT_COLOR_RGB)) \
    .then(lambda doc: doc_append(doc, BoxedEquationDefinition))
    # .then(lambda doc: doc_append(doc, NoEscape(r"\tolerance=10000 \pretolerance=10000")))
