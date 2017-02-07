import cv2
import numpy as np
from matplotlib import pyplot as plt


def ssd(A, B):
    squares = (A - B) ** 2
    return np.sum(squares)


def findTranslation(pixel_range, channel, im_color):
    minsumofsquarex = 99999999999999
    minsumofsquarey = 99999999999999
    height, width = im_color[:, :, 1].shape

    for i in range(-pixel_range, pixel_range):
        for j in range(-pixel_range, pixel_range):
            M = np.float32([[1, 0, i], [0, 1, j]])
            sd = np.zeros((height - (2 * pixel_range), width - (2 * pixel_range), 3), dtype=np.float)
            sd[:, :, 0] = im_color[pixel_range:-pixel_range, pixel_range:-pixel_range, 0]
            sd[:, :, channel] = cv2.warpAffine(
                im_color[pixel_range:-pixel_range, pixel_range:-pixel_range, channel], M,
                (width - (2 * pixel_range), height - (2 * pixel_range)))
            sumofsquarex = ssd(sd[:, :, 0], sd[:, :, channel])
            if sumofsquarex < minsumofsquarex:
                minsumofsquarex = sumofsquarex
                translation_x = i
                translation_y = j

    return translation_x, translation_y


def openImage(filepath):
    im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return im


def generateTwoImages(im, scale):
    im = cv2.resize(im, (int(im.shape[1] / scale), int(im.shape[0] / scale)), interpolation=cv2.INTER_AREA)
    im_to_align = im
    im = cv2.Canny(im, 100, 200)
    im = cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    im_to_align = cv2.normalize(im_to_align.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    # Find the width and height of the color image
    sz = im.shape
    print('image size:', sz)
    height = int(sz[0] / 3)
    width = sz[1]

    # Extract the three channels from the gray scale image
    # and merge the three channels into one color image
    im_bgr = np.zeros((height, width, 3), dtype=np.float)
    for i in range(0, 3):
        im_bgr[:, :, i] = im[i * height:(i + 1) * height, :]

    im_color = np.zeros((height, width, 3), dtype=np.float)
    im_color[:, :, 0] = im_bgr[:, :, 2]
    im_color[:, :, 1] = im_bgr[:, :, 1]
    im_color[:, :, 2] = im_bgr[:, :, 0]

    return im_color, im_to_align


def alignImage(im_color, im_to_align, x1, y1, x2, y2):
    print('x1: ', x1)
    print('y1: ', y1)
    print('x2: ', x2)
    print('y2: ', y2)
    aligned_translation = np.float32([[1, 0, x1], [0, 1, y1]])
    aligned_translation2 = np.float32([[1, 0, x2], [0, 1, y2]])

    height, width = im_color[:, :, 1].shape

    im_bgr_to_align = np.zeros((height, width, 3), dtype=np.float)
    for i in range(0, 3):
        im_bgr_to_align[:, :, i] = im_to_align[i * height:(i + 1) * height, :]

    im_color_to_align = np.zeros((height, width, 3), dtype=np.float)
    im_color_to_align[:, :, 0] = im_bgr_to_align[:, :, 2]
    im_color_to_align[:, :, 1] = im_bgr_to_align[:, :, 1]
    im_color_to_align[:, :, 2] = im_bgr_to_align[:, :, 0]

    aligned_image = np.zeros((height, width, 3), dtype=np.float)
    aligned_image[:, :, 0] = im_color_to_align[:, :, 0]
    aligned_image[:, :, 1] = cv2.warpAffine(im_color_to_align[:, :, 1], aligned_translation, (width, height))
    aligned_image[:, :, 2] = cv2.warpAffine(im_color_to_align[:, :, 2], aligned_translation2, (width, height))
    return aligned_image
