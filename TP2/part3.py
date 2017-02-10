import pylab as pl
from TP2.roipoly import roipoly



import cv2
from matplotlib import pyplot as plt
from TP2.imageAlignement import align_images, norm_image
from TP2.hybridImage import hybridImage, lowPass
from scipy import ndimage, misc
import numpy as np

apple = cv2.imread('apple.jpeg')
orange = cv2.imread('orange.jpeg')

arbitrary_value_1 = 2
arbitrary_value_2 = 2
cutoff_low = arbitrary_value_1
cutoff_high = arbitrary_value_2


for i in range(8):
    sigma = pow(cutoff_low ,i)
    low_pass_img = lowPass(apple, sigma)
    print("sigma: ",sigma, "i: ", i)

    low_pass_img = cv2.convertScaleAbs(low_pass_img)
    misc.imsave('part3/gaussian_pile_apple'+str(i)+'.png', low_pass_img)

for i in range(7):
    current_gaussian = cv2.imread('part3/gaussian_pile_apple' + str(i)+'.png')
    current_laplacian = cv2.imread('part3/gaussian_pile_apple' + str(i+1)+'.png') - current_gaussian
    misc.imsave('part3/laplacian_pile_apple'+str(i)+'.png', current_laplacian)

for i in range(8):
    sigma = pow(cutoff_low, i)
    low_pass_img = lowPass(orange, sigma)
    print("sigma: ", sigma, "i: ", i)

    low_pass_img = cv2.convertScaleAbs(low_pass_img)
    misc.imsave('part3/gaussian_pile_orange' + str(i) + '.png', low_pass_img)

for i in range(7):
    current_gaussian = cv2.imread('part3/gaussian_pile_orange' + str(i) + '.png')
    current_laplacian = cv2.imread('part3/gaussian_pile_orange' + str(i + 1) + '.png') - current_gaussian
    misc.imsave('part3/laplacian_pile_orange' + str(i) + '.png', current_laplacian)

GR = np.zeros(apple.shape)
for i in range(0, int(apple.shape[0]/2)):
    GR[:,i] = 1

for i in range(7):
    LA = cv2.imread('part3/laplacian_pile_apple' + str(i) + '.png')
    LB = cv2.imread('part3/laplacian_pile_orange' + str(i) + '.png')

    LS = np.multiply(GR,LA) + np.multiply((1-GR), LB)

    misc.imsave('part3/LS_' + str(i) + '.png', LS)

ls = np.zeros(apple.shape)
for i in range(7):
    current_ls = cv2.imread('part3/LS_' + str(i) + '.png')

    ls = ls + current_ls

last_gaussian_apple = cv2.imread('part3/gaussian_pile_apple7.png')
last_gaussian_orange = cv2.imread('part3/gaussian_pile_orange7.png')
ls = ls + np.multiply(last_gaussian_apple, GR) + np.multiply(last_gaussian_orange, (1-GR))
misc.imsave('part3/pommange.png', ls)

#More images


danny = cv2.imread('danny_amendola.png')
julian = cv2.imread('julian_edelman.png')

arbitrary_value_1 = 2
arbitrary_value_2 = 2
cutoff_low = arbitrary_value_1
cutoff_high = arbitrary_value_2


for i in range(8):
    sigma = pow(cutoff_low ,i)
    low_pass_img = lowPass(danny, sigma)
    print("sigma: ",sigma, "i: ", i)

    low_pass_img = cv2.convertScaleAbs(low_pass_img)
    misc.imsave('part3/gaussian_pile_danny'+str(i)+'.png', low_pass_img)

for i in range(7):
    current_gaussian = cv2.imread('part3/gaussian_pile_danny' + str(i)+'.png')
    current_laplacian = cv2.imread('part3/gaussian_pile_danny' + str(i+1)+'.png') - current_gaussian
    misc.imsave('part3/laplacian_pile_danny'+str(i)+'.png', current_laplacian)

for i in range(8):
    sigma = pow(cutoff_low, i)
    low_pass_img = lowPass(julian, sigma)
    print("sigma: ", sigma, "i: ", i)

    low_pass_img = cv2.convertScaleAbs(low_pass_img)
    misc.imsave('part3/gaussian_pile_julian' + str(i) + '.png', low_pass_img)

for i in range(7):
    current_gaussian = cv2.imread('part3/gaussian_pile_julian' + str(i) + '.png')
    current_laplacian = cv2.imread('part3/gaussian_pile_julian' + str(i + 1) + '.png') - current_gaussian
    misc.imsave('part3/laplacian_pile_julian' + str(i) + '.png', current_laplacian)

img = pl.ones(danny.shape)


# show the image
pl.imshow(img, interpolation='nearest', cmap="Greys")
pl.colorbar()
pl.title("left click: line segment         right click: close region")

# let user draw first ROI
ROI1 = roipoly(roicolor='r') #let user draw first ROI

# show the image with the first ROI
pl.imshow(img, interpolation='nearest', cmap="Greys")
pl.colorbar()
ROI1.displayROI()



# show the image with both ROIs and their mean values
pl.imshow(img, interpolation='nearest', cmap="Greys")
pl.colorbar()
[x.displayROI() for x in [ROI1]]
[x.displayMean(img) for x in [ROI1]]
pl.title('The ROI')
pl.show()

# show ROI masks
pl.imshow(ROI1.getMask(img),
          interpolation='nearest', cmap="Greys")
pl.title('ROI mask')
pl.show()

for i in range(0, int(danny.shape[0]/2)):
    GR[:,i] = 1

for i in range(7):
    LA = cv2.imread('part3/laplacian_pile_danny' + str(i) + '.png')
    LB = cv2.imread('part3/laplacian_pile_julian' + str(i) + '.png')

    LS = np.multiply(GR,LA) + np.multiply((1-GR), LB)

    misc.imsave('part3/LS_PATRIOTS_' + str(i) + '.png', LS)

ls = np.zeros(danny.shape)
for i in range(7):
    current_ls = cv2.imread('part3/LS_PATRIOTS_' + str(i) + '.png')

    ls = ls + current_ls

last_gaussian_danny = cv2.imread('part3/gaussian_pile_danny7.png')
last_gaussian_julian = cv2.imread('part3/gaussian_pile_julian7.png')
ls = ls + np.multiply(last_gaussian_danny, GR) + np.multiply(last_gaussian_julian, (1-GR))

misc.imsave('part3/Danian.png', ls)


