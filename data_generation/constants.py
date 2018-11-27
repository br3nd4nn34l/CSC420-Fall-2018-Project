import os

#region COLOR

COLOR_MODEL = "RGB"

BOX_COLOR_NAME = "BOXCOLOR"
BOX_COLOR_RGB = (0, 0, 255)  # BLUE

BOX_TEXT_COLOR_NAME = "BOXTEXTCOLOR"
BOX_TEXT_COLOR_RGB = (255, 0, 255)  # PURPLE

OLD_TEXT_COLOR_RGB = (0, 0, 0)  # BLACK
NEW_TEXT_COLOR_RGB = (255, 0, 0)  # RED

OLD_PAGE_COLOR_RGB = (255, 255, 255)  # WHITE
NEW_PAGE_COLOR_RGB = (0, 0, 0)  # BLACK

#endregion

#region PATHS

EQUATION_LIST_PATH = os.path.join("data_sources", "im2latex_formulas.lst")

#endregion