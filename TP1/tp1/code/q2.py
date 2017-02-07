import glob
from matplotlib import pyplot as plt
import scipy.misc
import cv2

from utils import alignImage, findTranslation, openImage, generateTwoImages, ssd
import numpy as np


def findTranslationMultiple(pixel_range, channel, im_color):
    minsumofsquarex = 99999999999999
    minsumofsquarey = 99999999999999
    im_copy = im_color
    current_translation = [0, 0]
    list = [32, 16, 8, 4, 2, 1]
    for scale in list:
        current_translation[0] = current_translation[0] * 2
        current_translation[1] = current_translation[1] * 2
        # resize les images
        im = cv2.resize(im_copy, (int(im_copy.shape[1] / scale), int(im_copy.shape[0] / scale)),
                        interpolation=cv2.INTER_AREA)
        im = cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        im = cv2.Canny(im, 100, 200)

        height = im.shape[0]
        width = im.shape[1]
        aligned_image = np.zeros((height, width, 3), dtype=np.float)
        aligned_translation = np.float32([[1, 0, current_translation[0]], [0, 1, current_translation[1]]])
        aligned_image[:, :, channel] = cv2.warpAffine(im[:, :, channel], aligned_translation, (width, height))
        plt.imshow(aligned_image)
        plt.show()
        for i in range(-pixel_range, pixel_range):
            for j in range(-pixel_range, pixel_range):
                M = np.float32([[1, 0, i], [0, 1, j]])
                sd = np.zeros((height - (2 * pixel_range), width - (2 * pixel_range), 3), dtype=np.float)
                sd[:, :, 0] = aligned_image[pixel_range:-pixel_range, pixel_range:-pixel_range, 0]
                sd[:, :, channel] = cv2.warpAffine(
                    aligned_image[pixel_range:-pixel_range, pixel_range:-pixel_range, channel], M,
                    (width - (2 * pixel_range), height - (2 * pixel_range)))
                sumofsquarex = ssd(sd[:, :, 0], sd[:, :, channel])
                if sumofsquarex < minsumofsquarex:
                    minsumofsquarex = sumofsquarex
                    translation_x = i
                    translation_y = j
        current_translation[0] = current_translation[0] + translation_x
        current_translation[1] = current_translation[1] + translation_y
        pixel_range = 5
        minsumofsquarex = 99999999999999

    return current_translation[0], current_translation[1]


path = "images/*.tif"
for fname in glob.glob(path):
    print(fname)
    im = openImage(fname)
    scale = 1
    im_color, im_to_align = generateTwoImages(im, scale)
    x_1, y_1 = findTranslationMultiple(15, 1, im_color)
    x_2, y_2 = findTranslationMultiple(15, 2, im_color)
    im_color, im_to_align = generateTwoImages(im, 1)
    print(x_1,y_1,x_2,y_2)

    aligned_image = alignImage(im_color, im_to_align, x_1, y_1, x_2, y_2)

    scipy.misc.imsave(fname.replace('images', 'resultats').replace('.tif', '.jpg'), aligned_image)
