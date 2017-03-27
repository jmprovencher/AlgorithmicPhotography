import cv2
import numpy as np
from numpy.linalg import inv
from math import floor, floor


def appliqueTransformation(img, H):
    rows, cols, color = img.shape
    upperLeft = [0, 0, 1]
    upperRight = [cols, 0, 1]
    lowerLeft = [0, rows, 1]
    lowerRight = [cols, rows, 1]
    inv_h = inv(H)

    newUpperLeft = np.dot(H, upperLeft)
    newUpperLeft /= newUpperLeft[2]
    newUpperLeft = newUpperLeft[0:2]

    newUpperRight = np.dot(H, upperRight)
    newUpperRight /= newUpperRight[2]
    newUpperRight = newUpperRight[0:2]

    newLowerLeft = np.dot(H, lowerLeft)
    newLowerLeft /= newLowerLeft[2]
    newLowerLeft = newLowerLeft[0:2]

    newLowerRight = np.dot(H, lowerRight)
    newLowerRight /= newLowerRight[2]
    newLowerRight = newLowerRight[0:2]

    corners = np.array((newUpperLeft, newUpperRight, newLowerLeft, newLowerRight))

    min_value = np.amin(corners, axis=0)
    max_value = np.amax(corners, axis=0)

    min_x = min_value[0]
    min_y = min_value[1]
    max_x = max_value[0]
    max_y = max_value[1]
    offset = [int(floor(min_x)),int(floor(min_y)),int(floor(max_x)),int(floor(max_y))]

    new_size = max_value - min_value

    new_image = np.zeros((floor(new_size[1]), floor(new_size[0]), 3), dtype='uint8')
    x, y = int(new_image.shape[1]), int(new_image.shape[0])
    for i in range(x):
        for j in range(y):
            coordinate = np.array((i + min_value[0], j + min_value[1], 1))
            pixel_in_original_image = np.dot(inv_h, coordinate)
            pixel_in_original_image /= pixel_in_original_image[2]
            if pixel_in_original_image[0] >= 0 and pixel_in_original_image[1] >= 0 and pixel_in_original_image[
                0] <= cols and pixel_in_original_image[1] <= rows:
                new_x = int(pixel_in_original_image[0])
                new_y = int(pixel_in_original_image[1])
                new_image[j][i] = img[new_y][new_x]

    print('applique premiere transform done')

    return new_image, offset
