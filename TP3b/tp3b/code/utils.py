from scipy.spatial import Delaunay
import numpy as np
import cv2


def getPointsFromFile(filepath):
    file = open(filepath, "r")
    file_array = file.readlines()

    list_to_append = []

    for line in file_array:
        current = line.split()
        current[0] = int(float(current[0]))
        current[1] = int(float(current[1]))
        list_to_append.append(current)

    return np.array(list_to_append)


def delaunay(points):
    tri = Delaunay(points)
    return tri

