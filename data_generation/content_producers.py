from pylatex import NoEscape, Section
from pylatex.lists import Enumerate, Itemize, Description

from faker import Faker

from data_generation.constants import EQUATION_LIST_PATH

from data_generation.generic_producers import UniformProducer, NormalProducer, \
    ListProducer, \
    StandardUniform, BoolCoinFlip


from data_generation.latex_commands import BoxedEquation

#region SEMANTICS

# Produces random latex equations with list
equation_list = list(open(EQUATION_LIST_PATH, "r").readlines())
EquationProducer = UniformProducer(0, len(equation_list))\
    .then(lambda x : min(int(x), len(equation_list) - 1))\
    .then(lambda ind: equation_list[ind])\
    .then(lambda eqn: BoxedEquation(fr"\({eqn}\)"))\
    .then(lambda eqn : NoEscape(eqn.dumps()))

fake = Faker()
def fake_words(num_words):
    return fake.sentence(nb_words=num_words, variable_nb_words=False)\
        .strip(".")\
        .split()

# Produces sentences with only words
WordSentenceProducer = NormalProducer(mu=14, sigma=2)\
    .then(lambda x : max(1, int(abs(x))))\
    .then(lambda num_words: fake_words(num_words))\
    .then(lambda words : " ".join(words))

def randomly_insert(lst, item):
    ind = int(len(item) * StandardUniform())
    return lst[:ind] + [item] + lst[ind:]

# Produces sentences with inline math
MathSentenceProducer = NormalProducer(mu=8, sigma=4)\
    .then(lambda x: max(1, int(abs(x))))\
    .then(lambda num_words: fake_words(num_words))\
    .then(lambda words : randomly_insert(words, EquationProducer()))\
    .then(lambda words : " ".join(words))

# Produces ending punctuation
EndPunctuationProducer = ListProducer(
    [".", "!", "?", "..."],
    [0.7, 0.05, 0.1, 0.15]
).then(lambda lst: lst[0])

# Produces tidbits (partial sentences, e.g. titles)
TidbitProducer = NormalProducer(mu=3, sigma=2)\
    .then(lambda x: max(1, int(abs(x))))\
    .then(lambda num_words : fake_words(num_words))\
    .then(lambda words : randomly_insert(words, EquationProducer()) if StandardUniform() < 0.1 else words)\
    .then(lambda words : " ".join(words))\
    .then(lambda tidbit : NoEscape(tidbit))

# Produces pure word or math sentences
SentenceProducer = ListProducer([WordSentenceProducer, MathSentenceProducer],
                                [0.8, 0.2])\
    .then(lambda lst : lst[0])\
    .then(lambda sentence : sentence if BoolCoinFlip() else sentence + EndPunctuationProducer())\
    .then(lambda sentence : NoEscape(sentence))

#endregion

#region REGIONS: SECTIONS, SUBSECTIONS, ETC

SectionProducer = TidbitProducer\
    .then(lambda tidbit : Section(tidbit, label=False))

#endregion

#region COLLECTIONS: PARAGRAPHS, LISTS, SE

# Produces paragraphs (contain words or math)
ParagraphProducer = NormalProducer(mu=5, sigma=2)\
    .then(lambda x: int(abs(x)))\
    .then(lambda num_sentences : [SentenceProducer() for i in range(num_sentences)])\
    .then(lambda sentences : " ".join(sentences))\
    .then(lambda paragraph : NoEscape(paragraph))

# To produce items for a list
ListItemProducer = ListProducer(
    [EquationProducer, TidbitProducer, SentenceProducer, ParagraphProducer],
    [0.1, 0.15, 0.45, 0.3]
).then(lambda lst : lst[0])

# To produce a list of list contents
ListContentProducer = NormalProducer(mu=3, sigma=3)\
    .then(lambda x : int(abs(x)))\
    .then(lambda num_items: [ListItemProducer() for i in range(num_items)])\

def itemize(contents):
    ret = Itemize()
    for item in contents:
        ret.add_item(item)
    return ret

ItemizeProducer = ListContentProducer\
    .then(lambda contents : itemize(contents))

def make_enumerate(contents):
    ret = Enumerate()
    for item in contents:
        ret.add_item(item)
    return ret

EnumerateProducer = ListContentProducer\
    .then(lambda contents : make_enumerate(contents))

def make_description(contents):
    ret = Description()
    for item in contents:
        label = TidbitProducer()
        ret.add_item(label, item)
    return ret

DescriptionProducer = ListContentProducer\
    .then(lambda contents : make_description(contents))

CollectionProducer = ListProducer(
    [ItemizeProducer, EnumerateProducer, DescriptionProducer]
).then(lambda lst : lst[0])
#endregion