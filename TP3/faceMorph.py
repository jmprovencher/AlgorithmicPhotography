#!/usr/bin/env python
from TP3.utils import delaunay, getPointsFromFile

import numpy as np
import cv2
import os
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize


def make_video(images, outvid=None, fps=5, size=None,
               is_color=True, format="MP42"):
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid


# Application de la transformation affine
def applyAffineTransform(src, srcTri, dstTri, size):
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


def morphTriangle(img1, img2, img, t1, t2, t, dissolve_frac):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    t1Rect = []
    t2Rect = []
    tRect = []

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    imgRect = (1.0 - dissolve_frac) * warpImage1 + dissolve_frac * warpImage2
    img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (1 - mask) + imgRect * mask


def morph(img1, img2, img1_pts, img2_pts, tri, warp_frac, dissolve_frac):
    img1 = np.float32(img1)
    img2 = np.float32(img2)

    imgMorph = np.zeros(img1.shape, dtype=img1.dtype)
    points = []
    # Moyenne pondérée des coordonnées de points
    for j in range(0, len(points1)):
        x = (1 - warp_frac) * points1[j][0] + warp_frac * points2[j][0]
        y = (1 - warp_frac) * points1[j][1] + warp_frac * points2[j][1]
        points.append((x, y))

    for element in tri.simplices:
        x = element[0]
        y = element[1]
        z = element[2]

        x = int(x)
        y = int(y)
        z = int(z)

        t1 = [points1[x], points1[y], points1[z]]
        t2 = [points2[x], points2[y], points2[z]]
        t = [points[x], points[y], points[z]]

        # Morph one triangle at a time.
        morphTriangle(img1, img2, imgMorph, t1, t2, t, dissolve_frac)
    return imgMorph.astype('u1')


if __name__ == '__main__':
    filename1 = "faces/regie.png"
    filename2 = "faces/nike.png"

    img1 = cv2.imread(filename1)
    img2 = cv2.imread(filename2)
    points1 = getPointsFromFile("points/regie.txt")
    points2 = getPointsFromFile("points/nike.txt")
    tri_points = (points1 + points2) / 2
    imagesArray = []
    tri = delaunay(tri_points)

    numberOfImages = 100

    for i in range(numberOfImages):
        warp_frac = i / numberOfImages
        dissolve_frac = i / numberOfImages
        imagesArray.append(morph(img1, img2, points1, points2, tri, warp_frac, dissolve_frac))

    height, width, layers = imagesArray[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('renike.mp4', fourcc, 30, (width, height))

    for image in imagesArray:
        out.write(image)

    cv2.destroyAllWindows()
    out.release()
