import numpy as np
import math
import scipy.misc as misc
import matplotlib.pyplot as plt
import cv2
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from TP2.imageAlignement import norm_image



def lowPass(imageMatrix, sigma):
    lowpass_img = cv2.GaussianBlur(imageMatrix, (201, 201), sigma)
    lowpass_img = lowpass_img.astype(np.int)
    return lowpass_img


def highPass(imageMatrix, sigma):
    lowpass_img = lowPass(imageMatrix, sigma)
    highpass_img = cv2.addWeighted(imageMatrix.astype(np.int), 1, lowpass_img, -1, 0)
    return highpass_img


def hybridImage(img1, img2, cutoff_low, cutoff_high):
    img2 = highPass(img2, cutoff_high)
    img1 = lowPass(img1, cutoff_low)
    print(img2)
    print(img1)
    return cv2.addWeighted(img1, 1, img2, 0.5, 0)
