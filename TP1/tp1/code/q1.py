import glob

import scipy.misc

from utils import openImage,findTranslation,alignImage, generateTwoImages

path = "images/*.jpg"
for fname in glob.glob(path):
    print(fname)
    im = openImage(fname)
    im_color, im_to_align = generateTwoImages(im, 1)

    x_1, y_1 = findTranslation(15, 1, im_color)
    x_2, y_2 = findTranslation(15, 2, im_color)

    aligned_image = alignImage(im_color, im_to_align, x_1, y_1, x_2, y_2)
    scipy.misc.imsave(fname.replace('images', 'resultats'), aligned_image)
