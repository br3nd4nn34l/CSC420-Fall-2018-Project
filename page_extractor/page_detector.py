import cv2
import numpy as np
import math
import itertools
from bresenham import bresenham


def get_angle(line1, line2):
    """
    Given 2 lines represented as 2 coordinates each, return the
    angle at the intersection of the 2 lines (in degrees).

    The function assumes that the 2 lines intersect.
    """
    (Ax1, Ay1), (Ax2, Ay2) = line1
    (Bx1, By1), (Bx2, By2) = line2
    angle1 = math.atan2(Ay1 - Ay2,
                        Ax1 - Ax2)
    angle2 = math.atan2(By1 - By2,
                        Bx1 - Bx2)
    return math.degrees(angle1 - angle2)


def line_intersection(line1, line2, img_shape):
    """
    Gets the point where the 2 lines intersects.
    Returns None if the lines are parallel, 0 if invalid intersection
    """
    (Ax1, Ay1), (Ax2, Ay2) = line1
    (Bx1, By1), (Bx2, By2) = line2
    xdiff = (Ax1 - Ax2, Bx1 - Bx2)
    ydiff = (Ay1 - Ay2, By1 - By2)

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)

    # The 2 lines are very close together which means they are probably
    # referencing the same edge. Only 1 of the lines should be kept.
    if (abs(Ax1 - Bx1) < 25 and abs(Ay1 - By1) < 25 and
                abs(Ax2 - Bx2) < 25 and abs(Ay2 - By2) < 25):
        return 0

    # No intersection between lines. Generally doesn't happen unless
    # the lines are completely parallel. This algorithm assumes that
    # the lines are infinitely long.
    if div == 0:
        # If the 2 lines have less than 10 pixel difference, then they are probably referencing
        # the same edge. return 0 so that one of the two lines gets removed.
        if minimum_distance(line1, (Bx1, By1)) < 10:
            return 0

        return None

    # Get the angle of the line intersection
    angle = abs(get_angle(line1, line2))
    angle = min(angle, abs(angle - 180))

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    # Make sure the intersection happens within the image bounds
    if not line_within_bounds((int(x), int(y)), img_shape):
        return None

    # Eliminate line intersections that have a large or small angle
    if angle > 115 or angle < 65:
        return 0

    return int(x), int(y)


def minimum_distance(line, pt):
    """
    Returns the minimum distance between the point and line.
    """
    pt1, pt2 = np.array(line[0]), np.array(line[1])
    pt3 = np.array(pt)

    return np.abs(np.cross(pt2 - pt1, pt1 - pt3) / np.linalg.norm(pt2 - pt1))


def get_non_zeros(img):
    """
    Given an img, get the coordinates that are not 0.
    """
    coords = []
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i, j] != 0:
                coords.append((j, i))
    return coords


def no_duplicate_check(pts):
    """
    Make sure none of the 4 coordinates are duplicates
    """
    for i in range(4):
        for j in range(i + 1, 4):
            if pts[i] == (pts[j][1], pts[j][0]):
                return False
    return True


def find_quadrilateral(points, canny):
    # Store points along with its score
    line_coords = itertools.permutations(points, 2)

    lines = {}
    canny_pts = get_non_zeros(canny)

    for l in line_coords:
        # Get all the pixel coordinates between the 2 line coordinates that
        # intersect with canny edge image
        overlap_pixels = len(set(list(bresenham(l[0][0], l[0][1], l[1][0], l[1][1]))).intersection(canny_pts))
        if overlap_pixels > 0:
            lines[l] = overlap_pixels

    # Get all permutations of the 4 coordinates
    possible_outline = itertools.permutations(points, 4)
    quad = ((0, 0) * 4)
    max_length = 0

    # Go through all 4 coordinate combinations and find the best match
    for l in possible_outline:
        # Skip over if not a valid quadrilateral.
        if (l[0:2] not in lines or l[1:3] not in lines or
                    l[2:4] not in lines or (l[0], l[3]) not in lines):
            continue

        # Count the total number of pixels the quadrilateral matches with the
        # canny edge image and pick the quadrilateral if it is the highest match.
        count = lines[l[0:2]] + lines[l[1:3]] + lines[l[2:4]] + lines[(l[0], l[3])]
        if count > max_length and no_duplicate_check(l):
            quad = l
            max_length = count

    return np.array([quad[0], quad[1], quad[2], quad[3]])


def line_within_bounds(point, img_shape):
    """
    Return true if the point given is within the bounds of the image.
    """
    return 0 <= point[0] < img_shape[1] and 0 <= point[1] < img_shape[0]


def get_canny(I, min_threshold=300, max_threshold=700):
    """
    Get the canny edge image given an image and min/max thresholds.
    """
    return cv2.Canny(I, min_threshold, max_threshold)


def get_hough_lines(edges, img, k=20):
    """
    Use hough transformation to find edges in a given image.

    returns the K best matched lines.
    """
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 50)
    max_size = int(np.sqrt(np.square(img.shape[0]) + np.square(img.shape[1])))
    hough_lines = []
    canny_pts = get_non_zeros(edges)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + max_size * (-b))
        y1 = int(y0 + max_size * (a))
        x2 = int(x0 - max_size * (-b))
        y2 = int(y0 - max_size * (a))

        intersection = ((x1, y1), (x2, y2))

        hough_line_pixels = list(bresenham(intersection[0][0], intersection[0][1],
                           intersection[1][0], intersection[1][1]))
        intersection_matches = len(set(hough_line_pixels).intersection(canny_pts))
        hough_lines.append((intersection, intersection_matches))

    hough_lines = sorted(hough_lines, key=lambda x: x[1], reverse=True)
    return hough_lines[:k]


def get_all_intersections(lines, img_shape, new_img):
    """
    Get all line intersections that meet the set conditions:

    - Angle must be between 65 degrees and 115 degrees
    - Intersection must be within the bounds of the image.
    """
    all_intersections = []
    lines_to_remove = set()
    # Iterate through all line pairs searching for intersections
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            # Get the intersection point of the 2 lines
            point = line_intersection(lines[i][0], lines[j][0], img_shape)
            if point:
                all_intersections.append((point, lines[i], lines[j]))
            # If point gets value 0, it means the 2 lines are referencing the same
            # edge and the one with fewer overlaps with the canny edge image
            # should be removed.
            elif point == 0:
                if lines[i][1] > lines[j][1]:
                    lines_to_remove.add(lines[j])
                else:
                    lines_to_remove.add(lines[i])

    # Update lines and intersection points to remove outliers
    lines = list(set(lines).difference(lines_to_remove))

    # Draw the lines to the image. [ONLY FOR TESTING]
    for i in lines:
        cv2.line(new_img, i[0][0], i[0][1], (0, 0, 255), 2)

    intersections = []
    for intersection in all_intersections:
        # Only keep intersections for lines that are kept
        if intersection[1] in lines and intersection[2] in lines:
            intersections.append(intersection[0])

            # Draw the intersection points to the image. [ONLY FOR TESTING]
            cv2.line(new_img, intersections[-1], intersections[-1], (0, 255, 0), 10)

    return intersections, lines, new_img


def get_page_corners(img):
    """
    Returns the 4 corner coordinates of the paper in the image.
    """
    edges = get_canny(img)
    lines = get_hough_lines(edges, img)
    intersections, lines, new_img = get_all_intersections(lines, img.shape, np.copy(img))
    quads = find_quadrilateral(intersections, edges)
    return quads


if __name__ == '__main__':
    img = cv2.imread('images/paper.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    edges = get_canny(img)
    lines = get_hough_lines(edges, img)
    intersections, lines, new_img = get_all_intersections(lines, img.shape, np.copy(img))
    quads = find_quadrilateral(intersections, edges)
    quads = np.array(quads).reshape(-1, 1, 2)
    # print(quads)
    # cv2.polylines(img, quads, True, (0, 0, 255), 15)
    # cv2.imshow("original", img)
    # cv2.imshow("canny edges", edges)
    # cv2.imshow("original + edges + intersections", new_img)
    # cv2.waitKey(0)
    import page_transformation

    img = page_transformation.homography(quads, img)
    cv2.imshow("original + edges + intersections", img)
    # cv2.imwrite("final_result.jpg", img)
    cv2.waitKey(0)

