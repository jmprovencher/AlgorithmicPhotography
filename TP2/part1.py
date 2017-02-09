import cv2
from matplotlib import pyplot as plt
from TP2.imageAlignement import align_images, norm_image
from TP2.hybridImage import hybridImage
from scipy import ndimage, misc
import numpy

albert = cv2.imread('justin_trudeau.jpg', cv2.IMREAD_GRAYSCALE)
marilyn = cv2.imread('jeff_lalonde.jpg', cv2.IMREAD_GRAYSCALE)

arbitrary_value_1 = 5
arbitrary_value_2 = 60
cutoff_low = arbitrary_value_1
cutoff_high = arbitrary_value_2

albert, marilyn = align_images(albert, marilyn)
hybrid_img = hybridImage(marilyn, albert, cutoff_low, cutoff_high)

hybrid_img = cv2.convertScaleAbs(hybrid_img)
misc.imsave('hybrid5.png', hybrid_img)

plt.imshow(hybrid_img, cmap='gray')
plt.title('hybrid image')
plt.show()
