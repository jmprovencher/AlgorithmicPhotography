import scipy.misc
import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils import alignImage, findTranslation, openImage, generateTwoImages

fname_1_red = "images_personnelles/1_1.jpg"
fname_1_green = "images_personnelles/1_2.jpg"
fname_1_blue = "images_personnelles/1_3.jpg"
im_1_red = cv2.imread(fname_1_red)
im_1_red = cv2.normalize(im_1_red.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
red = im_1_red[:, :, 2]
im_1_green = cv2.imread(fname_1_green)
im_1_green = cv2.normalize(im_1_green.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
green = im_1_green[:, :, 1]

im_1_blue = cv2.imread(fname_1_blue)
im_1_blue = cv2.normalize(im_1_blue.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
blue = im_1_blue[:, :, 0]
height = im_1_blue.shape[0]
width = im_1_blue.shape[1]
im_color = np.zeros((height, width, 3), dtype=np.float)

im_color[:, :, 2] = blue
im_color[:, :, 1] = green
im_color[:, :, 0] = red

im_to_align = np.zeros((height, width, 3), dtype=np.float)

x1, y1 = findTranslation(15, 1, im_color)
x2, y2 = findTranslation(15, 2, im_color)

print('image 1')
print(x1, y1)
print(x2, y2)

aligned_translation = np.float32([[1, 0, x1], [0, 1, y1]])
aligned_translation2 = np.float32([[1, 0, x2], [0, 1, y2]])
aligned_image = np.zeros((height, width, 3), dtype=np.float)
aligned_image[:, :, 0] = im_color[:, :, 0]
aligned_image[:, :, 1] = cv2.warpAffine(im_color[:, :, 1], aligned_translation, (width, height))
aligned_image[:, :, 2] = cv2.warpAffine(im_color[:, :, 2], aligned_translation2, (width, height))

scipy.misc.imsave(fname_1_red.replace('images_personnelles', 'resultats'), aligned_image)

fname_1_red = "images_personnelles/2_1.jpg"
fname_1_green = "images_personnelles/2_2.jpg"
fname_1_blue = "images_personnelles/2_3.jpg"
im_1_red = cv2.imread(fname_1_red)
im_1_red = cv2.normalize(im_1_red.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
red = im_1_red[:, :, 2]
im_1_green = cv2.imread(fname_1_green)
im_1_green = cv2.normalize(im_1_green.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
green = im_1_green[:, :, 1]

im_1_blue = cv2.imread(fname_1_blue)
im_1_blue = cv2.normalize(im_1_blue.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
blue = im_1_blue[:, :, 0]
height = im_1_blue.shape[0]
width = im_1_blue.shape[1]
im_color = np.zeros((height, width, 3), dtype=np.float)

im_color[:, :, 2] = blue
im_color[:, :, 1] = green
im_color[:, :, 0] = red

im_to_align = np.zeros((height, width, 3), dtype=np.float)

x1, y1 = findTranslation(15, 1, im_color)
x2, y2 = findTranslation(15, 2, im_color)

print('image 2')
print(x1, y1)
print(x2, y2)

aligned_translation = np.float32([[1, 0, x1], [0, 1, y1]])
aligned_translation2 = np.float32([[1, 0, x2], [0, 1, y2]])
aligned_image = np.zeros((height, width, 3), dtype=np.float)
aligned_image[:, :, 0] = im_color[:, :, 0]
aligned_image[:, :, 1] = cv2.warpAffine(im_color[:, :, 1], aligned_translation, (width, height))
aligned_image[:, :, 2] = cv2.warpAffine(im_color[:, :, 2], aligned_translation2, (width, height))

scipy.misc.imsave(fname_1_red.replace('images_personnelles', 'resultats'), aligned_image)

fname_1_red = "images_personnelles/3_1.jpg"
fname_1_green = "images_personnelles/3_2.jpg"
fname_1_blue = "images_personnelles/3_3.jpg"
im_1_red = cv2.imread(fname_1_red)
im_1_red = cv2.normalize(im_1_red.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
red = im_1_red[:, :, 2]
im_1_green = cv2.imread(fname_1_green)
im_1_green = cv2.normalize(im_1_green.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
green = im_1_green[:, :, 1]

im_1_blue = cv2.imread(fname_1_blue)
im_1_blue = cv2.normalize(im_1_blue.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
blue = im_1_blue[:, :, 0]
height = im_1_blue.shape[0]
width = im_1_blue.shape[1]
im_color = np.zeros((height, width, 3), dtype=np.float)

im_color[:, :, 2] = blue
im_color[:, :, 1] = green
im_color[:, :, 0] = red

im_to_align = np.zeros((height, width, 3), dtype=np.float)

x1, y1 = findTranslation(15, 1, im_color)
x2, y2 = findTranslation(15, 2, im_color)

print('image 3')
print(x1, y1)
print(x2, y2)

aligned_translation = np.float32([[1, 0, x1], [0, 1, y1]])
aligned_translation2 = np.float32([[1, 0, x2], [0, 1, y2]])
aligned_image = np.zeros((height, width, 3), dtype=np.float)
aligned_image[:, :, 0] = im_color[:, :, 0]
aligned_image[:, :, 1] = cv2.warpAffine(im_color[:, :, 1], aligned_translation, (width, height))
aligned_image[:, :, 2] = cv2.warpAffine(im_color[:, :, 2], aligned_translation2, (width, height))

scipy.misc.imsave(fname_1_red.replace('images_personnelles', 'resultats'), aligned_image)
