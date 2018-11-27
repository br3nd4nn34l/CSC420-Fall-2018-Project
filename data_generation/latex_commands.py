from pylatex.base_classes import CommandBase
from pylatex import UnsafeCommand, Package, NoEscape

from data_generation.constants import BOX_COLOR_NAME, BOX_TEXT_COLOR_NAME

class BoxedEquationCommand(CommandBase):
    _latex_name = "boxedEquation"
    packages = [Package("color")]

def BoxedEquation(equation):
    return BoxedEquationCommand(NoEscape(equation))

boxed_equation_code = fr"""\colorbox{{{BOX_COLOR_NAME}}}{{\textcolor{{{BOX_TEXT_COLOR_NAME}}}{{#1}}}}"""

BoxedEquationDefinition = UnsafeCommand("newcommand", r"\boxedEquation", options=1,
                                        extra_arguments=boxed_equation_code)