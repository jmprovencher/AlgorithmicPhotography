import cv2
import numpy as np
from numpy.linalg import inv
from math import floor
from TP4.appliqueTransformation import appliqueTransformation
from TP4.gatherPoints import gatherPoints


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
    return H

def calculerHomographieModif(im1_pts, im2_pts):
    A = []

    for i in range(0, 4):
        x, y = im1_pts[i,0], im1_pts[i,1]
        u, v = im2_pts[i,0], im2_pts[i,1]
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.asarray(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    H = L.reshape(3, 3)
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
