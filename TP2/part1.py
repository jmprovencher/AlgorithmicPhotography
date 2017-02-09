import cv2
from matplotlib import pyplot as plt
from TP2.imageAlignement import align_images, norm_image
from TP2.hybridImage import hybridImage
from scipy import ndimage, misc
import numpy as np

albert = cv2.imread('justin_trudeau.jpg', cv2.IMREAD_GRAYSCALE)
marilyn = cv2.imread('jeff_lalonde.jpg', cv2.IMREAD_GRAYSCALE)

arbitrary_value_1 = 5
arbitrary_value_2 = 60
cutoff_low = arbitrary_value_1
cutoff_high = arbitrary_value_2

albert, marilyn = align_images(albert, marilyn)
hybrid_img, low_pass_img, high_pass_img = hybridImage(marilyn, albert, cutoff_low, cutoff_high)

hybrid_img = cv2.convertScaleAbs(hybrid_img)
misc.imsave('hybrid5.png', hybrid_img)

fig = plt.figure()
a=fig.add_subplot(2,3,1)
imgplot = plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(albert)))))
a.set_title('TF Justin Trudeau Originale')
a=fig.add_subplot(2,3,2)
imgplot2 = plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(marilyn)))))
a.set_title('TF Jean-Francois Lalonde Originale')
a=fig.add_subplot(2,3,3)
imgplot3 = plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(low_pass_img)))))
a.set_title('TF Jean-François Lalonde Low pass')
a=fig.add_subplot(2,3,4)
imgplot3 = plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(high_pass_img)))))
a.set_title('TF Justin Trudeau High pass')
a=fig.add_subplot(2,3,5)
imgplot3 = plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(hybrid_img)))))
a.set_title('TF Hybrid Image')
plt.show()
