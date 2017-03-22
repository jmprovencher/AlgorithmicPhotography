import cv2
import numpy as np
from numpy.linalg import inv
from math import floor, ceil
from TP4.appliqueTransformation import appliqueTransformation


def calculerHomographie(im1_pts, im2_pts):
    A = []
    for i in range(0, 4):
        x, y = im1_pts[i][0], im1_pts[i][1]
        u, v = im2_pts[i][0], im2_pts[i][1]
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.asarray(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    H = L.reshape(3, 3)
    print(H)
    return H


def getPointsFromFile(filepath):
    file = open(filepath, "r")
    file_array = file.readlines()
    list_to_append = []

    for line in file_array:
        xy = line.strip().split(",")
        point = []
        point.append(float(xy[0]))
        point.append(float(xy[1]))
        list_to_append.append(point)

    return list_to_append


im1_pts = getPointsFromFile("pts_serie1/pts1_12.txt")
im2_pts = getPointsFromFile("pts_serie1/pts2_12.txt")
print(im1_pts)
print(im2_pts)
H = calculerHomographie(im1_pts,im2_pts)
img = cv2.imread('images/1- PartieManuelle/Serie1/IMG_2415.JPG', 1)

transformed_image = appliqueTransformation(img, H)
cv2.imshow('img', transformed_image)
cv2.imwrite('JFLalonde_transformed.png',transformed_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
