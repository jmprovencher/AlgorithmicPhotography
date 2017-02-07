import numpy
import math
import scipy.misc as misc
import matplotlib.pyplot as plt
import cv2
from numpy.fft import fft2, ifft2, fftshift, ifftshift


def scaleSpectrum(A):
    return numpy.real(numpy.log10(numpy.absolute(A) + numpy.ones(A.shape)))


def makeGaussianFilter(numRows, numCols, sigma, highPass=True):
    centerI = int(numRows / 2) + 1 if numRows % 2 == 1 else int(numRows / 2)
    centerJ = int(numCols / 2) + 1 if numCols % 2 == 1 else int(numCols / 2)

    def gaussian(i, j):
        coefficient = math.exp(-1.0 * ((i - centerI) ** 2 + (j - centerJ) ** 2) / (2 * sigma ** 2))
        return 1 - coefficient if highPass else coefficient

    return numpy.array([[gaussian(i, j) for j in range(numCols)] for i in range(numRows)])


def lowPass(imageMatrix, sigma):
    n, m = imageMatrix.shape
    return filterDFT(imageMatrix, makeGaussianFilter(n, m, sigma, highPass=False))


def filterDFT(imageMatrix, filterMatrix):
    shiftedDFT = fftshift(fft2(imageMatrix))
    filteredDFT = shiftedDFT * filterMatrix
    return ifft2(ifftshift(filteredDFT))

def highPass(imageMatrix, sigma):
   n,m = imageMatrix.shape
   return filterDFT(imageMatrix, makeGaussianFilter(n, m, sigma, highPass=True))


def hybridImage(img1, img2, cutoff_low, cutoff_high):

    img2 = highPass(img2, cutoff_high)
    img1 = lowPass(img1, cutoff_low)

    return numpy.real(img2 + img1)
