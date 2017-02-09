import cv2
from matplotlib import pyplot as plt
from TP2.imageAlignement import align_images, norm_image
from TP2.hybridImage import hybridImage, lowPass
from scipy import ndimage, misc
import numpy as np

lincoln = cv2.imread('hybrid5.png', cv2.IMREAD_GRAYSCALE)

arbitrary_value_1 = 2
arbitrary_value_2 = 2
cutoff_low = arbitrary_value_1
cutoff_high = arbitrary_value_2


for i in range(8):
    sigma = pow(cutoff_low ,i)
    low_pass_img = lowPass(lincoln, sigma)
    print("sigma: ",sigma, "i: ", i)

    low_pass_img = cv2.convertScaleAbs(low_pass_img)
    misc.imsave('gaussian_pile_custom'+str(i)+'.png', low_pass_img)

for i in range(7):
    current_gaussian = cv2.imread('gaussian_pile_custom' + str(i)+'.png', cv2.IMREAD_GRAYSCALE)
    current_laplacian = cv2.imread('gaussian_pile_custom' + str(i+1)+'.png', cv2.IMREAD_GRAYSCALE) - current_gaussian
    misc.imsave('laplacian_pile_custom'+str(i)+'.png', current_laplacian)
