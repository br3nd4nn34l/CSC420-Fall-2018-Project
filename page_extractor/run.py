import cv2
from page_detector import get_page_corners
from page_transformation import homography
import matplotlib.pyplot as plt


def get_page(img_path):
    """
    Get the page from an image.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    corners = get_page_corners(img)
    new_img = homography(corners, img)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img, cmap='gray')
    ax[1].imshow(new_img, cmap='gray')
    plt.show()


if "__main__" == __name__:
    get_page(img_path="images/paper.jpg")
