import cv2
import numpy as np
import math
import itertools


def get_angle(line1, line2):
    angle1 = math.atan2(line1[0][1] - line1[1][1],
                        line1[0][0] - line1[1][0])
    angle2 = math.atan2(line2[0][1] - line2[1][1],
                        line2[0][0] - line2[1][0])
    return math.degrees(angle1 - angle2)


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    # Eliminate line intersections that have a large or small angle
    angle = abs(get_angle(line1, line2))
    angle = min(angle, abs(angle - 180))
    if angle > 115 or angle < 65:
        return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return int(x), int(y)


# TODO
def find_quadrilateral(points):
    iterations = itertools.permutations(points, 4)

    counter = 0
    for i in iterations:
        tl, tr, bl, br = i
        if not line_intersection((tl, tr), (bl, br)):
            counter += 1
    print(counter)

def line_with_bounds(point, img_shape):
    return 0 <= point[0] < img_shape[1] and 0 <= point[1] < img_shape[0]


def get_canny(I, min_threshold=400, max_threshold=700):
    return cv2.Canny(I, min_threshold, max_threshold)


def get_hough_lines(edges, I):
    # minLineLength = 0
    # maxLineGap = 0
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=minLineLength, maxLineGap=maxLineGap)
    max_size = int(np.sqrt(np.square(I.shape[0]) + np.square(I.shape[1])))
    hough_lines = []
    for line in lines:
        # for x1, y1, x2, y2 in line:
        #     cv2.line(I, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # x1, y1, x2, y2 = line[0]

        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = min(max_size, max(0, int(x0 + max_size * (-b))))
        y1 = min(max_size, max(0, int(y0 + max_size * (a))))
        x2 = min(max_size, max(0, int(x0 - max_size * (-b))))
        y2 = min(max_size, max(0, int(y0 - max_size * (a))))
        hough_lines.append([(x1, y1), (x2, y2)])
        cv2.line(I, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return I, hough_lines


def get_all_intersections(lines, img_shape, new_img):
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            point = line_intersection(lines[i], lines[j])
            if point and line_with_bounds(point, img_shape):
                intersections.append(point)
                cv2.line(new_img, intersections[-1], intersections[-1], (0, 255, 0), 10)
    print(len(intersections))
    return intersections, new_img


if __name__ == '__main__':
    img = cv2.imread('images/2.jpg')
    edges = get_canny(img)
    new_img, lines = get_hough_lines(edges, np.copy(img))
    intersections, new_I = get_all_intersections(lines, img.shape, new_img)
    # find_quadrilateral takes too long to run
    # find_quadrilateral(intersections)
    cv2.imshow("original", img)
    cv2.imshow("canny edges", edges)
    cv2.imshow("original + edges + intersections", new_img)
    cv2.waitKey(0)
