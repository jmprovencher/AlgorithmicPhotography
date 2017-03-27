from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter, maximum_filter
import numpy as np
import cv2
import math
from scipy.ndimage.interpolation import map_coordinates
from TP4.appliqueTransformation import appliqueTransformation
from TP4.calculerHomographie import calculerHomographieModif
from TP4.drawPoints import drawPoints

from scipy.spatial.distance import cdist

# img0 = cv2.imread('images/2- PartieAutomatique/Serie1/goldengate-00.png', 1)
# img1 = cv2.imread('images/2- PartieAutomatique/Serie1/goldengate-01.png', 1)
# img2 = cv2.imread('images/2- PartieAutomatique/Serie1/goldengate-02.png', 1)
# img3 = cv2.imread('images/2- PartieAutomatique/Serie1/goldengate-03.png', 1)
# img4 = cv2.imread('images/2- PartieAutomatique/Serie1/goldengate-04.png', 1)
# img5 = cv2.imread('images/2- PartieAutomatique/Serie1/goldengate-05.png', 1)

# img0 = cv2.imread('images/2- PartieAutomatique/Serie2/IMG_2415.JPG', 1)
# img1 = cv2.imread('images/2- PartieAutomatique/Serie2/IMG_2416.JPG', 1)
# img2 = cv2.imread('images/2- PartieAutomatique/Serie2/IMG_2417.JPG', 1)
# img3 = cv2.imread('images/2- PartieAutomatique/Serie2/IMG_2418.JPG', 1)

img0 = cv2.imread('images/2- PartieAutomatique/Serie3/IMG_2434.JPG', 1)
img1 = cv2.imread('images/2- PartieAutomatique/Serie3/IMG_2435.JPG', 1)
img2 = cv2.imread('images/2- PartieAutomatique/Serie3/IMG_2436.JPG', 1)
img3 = cv2.imread('images/2- PartieAutomatique/Serie3/IMG_2466.JPG', 1)
img4 = cv2.imread('images/2- PartieAutomatique/Serie3/IMG_2467.JPG', 1)
img5 = cv2.imread('images/2- PartieAutomatique/Serie3/IMG_2468.JPG', 1)


# img0 = cv2.imread('plt0.jpg', 1)
# img1 = cv2.imread('plt1.jpg', 1)
# img2 = cv2.imread('plt2.jpg', 1)


# Convertir img en grayscale
def transformRGBToGray(im):
    return 0.299 * im[..., 2] + 0.587 * im[..., 1] + 0.114 * im[..., 0]


# detecteur harris
def harris_detector(im, number=512, bord=40):
    direction_x = gaussian_filter1d(gaussian_filter1d(im.astype(np.float32), 1.0, 0, 0), 1.0, 1, 1)
    direction_y = gaussian_filter1d(gaussian_filter1d(im.astype(np.float32), 1.0, 1, 0), 1.0, 0, 1)
    h = (gaussian_filter(direction_x ** 2, 1.5, 0) * gaussian_filter(direction_y ** 2, 1.5, 0) - gaussian_filter(
        direction_x * direction_y, 1.5,
        0) ** 2) / (
            gaussian_filter(direction_x ** 2, 1.5, 0) + gaussian_filter(direction_y ** 2, 1.5, 0) + 1e-8)
    h[:bord, :], h[-bord:, :], h[:, :bord], h[:, -bord:] = 0, 0, 0, 0
    h = h * (h == maximum_filter(h, (8, 8)))
    directions = np.argsort(h.flatten())[::-1][:number]
    return np.vstack((directions % im.shape[0:2][1], directions / im.shape[0:2][1], h.flatten()[directions])).transpose()


# obtenir descripteur image
def extraireDescripteur(img, harris, rayon=8):
    y, x = 4 * np.mgrid[-rayon:rayon + 1, -rayon:rayon + 1]
    desc = np.zeros((2 * rayon + 1, 2 * rayon + 1, harris.shape[0]), dtype=float)
    for i in range(harris.shape[0]):
        patch = map_coordinates(img, [harris[i, 1] + y, harris[i, 0] + x], prefilter=False)
        desc[..., i] = (patch - patch.mean()) / patch.std()
    return desc


# obtenir correspondances entre deux descripteurs
def matchTwoDescriptor(d1, d2):
    height, width, n = d1.shape[0:3]
    distance = cdist((d1.reshape((width ** 2, n))).T, (d2.reshape((height ** 2, n))).T)
    distance_sorted_list = np.argsort(distance, 1)[:, 0]
    ratio = distance[np.r_[0:n], distance_sorted_list] / distance[np.r_[0:n], np.argsort(distance, 1)[:, 1]].mean()
    return np.hstack([np.argwhere(ratio < 0.5), distance_sorted_list[np.argwhere(ratio < 0.5)]]).astype(int)


# convertir en données homogènes
def convertToHomogeneous(in_data):
    if in_data.shape[0] == 3:
        homo_data = np.zeros_like(in_data)
        for i in range(3):
            homo_data[i, :] = in_data[i, :] / in_data[2, :]
    elif in_data.shape[0] == 2:
        homo_data = np.vstack((in_data, np.ones((1, in_data.shape[1]), dtype=in_data.dtype)))
    return homo_data


def RANSAC(points, threshold=0.5, max=100, confiance=0.95):
    i = 0
    h = None
    c = 0
    l = None
    while i < max:
        tempd, temps = np.matrix(np.copy(points)), np.copy(points)
        np.random.shuffle(temps)  # points aléatoires
        temps = np.matrix(temps)[0:4]

        homography = calculerHomographieModif(temps[:, 0:2], temps[:, 2:])
        erreur = np.sqrt((np.array(
            np.array(
                convertToHomogeneous((homography * convertToHomogeneous(tempd[:, 0:2].transpose())))[0:2, :]) - tempd[:,
                                                                                                                2:].transpose()) ** 2).sum(
            0))
        if (erreur < threshold).sum() > c:
            h = homography
            c = (erreur < threshold).sum()
            l = np.argwhere(erreur < threshold)
            p = float(c) / points.shape[0]
            max = math.log(1 - confiance) / math.log(1 - (p ** 4))
        i += 1
    return h, l


# ANMS, algorithme suppresseur de points
def pointSupressor(positions, best=80):
    matrix = []
    x = 0
    y = 0
    while x < len(positions):
        minimalPoint = 99999999
        xi = positions[x][0]
        yi = positions[x][1]
        while y < len(positions):
            xj, yj = positions[y][0], positions[y][1]
            if (xi != xj and yi != yj) and positions[x][2] < 0.9 * positions[y][2]:
                distancePoint = distance(xi, yi, xj, yj)
                if distancePoint < minimalPoint:
                    minimalPoint = distancePoint
            y += 1
        matrix.append([xi, yi, minimalPoint])
        x += 1
        y = 0
    matrix.sort(key=lambda x: x[2])
    matrix = matrix[0:best]
    return np.array(matrix)


def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)



image0, image1, image2, image3, image4, image5 = transformRGBToGray(img0), transformRGBToGray(img1), transformRGBToGray(img2), transformRGBToGray(
    img3), transformRGBToGray(img4), transformRGBToGray(img5)
points0, points1, points2, points3, points4, points5 = harris_detector(image0, number=500), harris_detector(image1, number=500), harris_detector(
    image2, number=500), harris_detector(image3, number=500), harris_detector(image4, number=500), harris_detector(image5, number=500)
points2_anms = pointSupressor(points2)
points3_anms = pointSupressor(points3)

matches23 = matchTwoDescriptor(extraireDescripteur(image2, points2), extraireDescripteur(image3, points3))
matches12 = matchTwoDescriptor(extraireDescripteur(image1, points1), extraireDescripteur(image2, points2))
matches01 = matchTwoDescriptor(extraireDescripteur(image0, points0), extraireDescripteur(image1, points1))

matches43 = matchTwoDescriptor(extraireDescripteur(image4, points4), extraireDescripteur(image3, points3))
matches54 = matchTwoDescriptor(extraireDescripteur(image5, points5), extraireDescripteur(image4, points4))

H23 = RANSAC(np.matrix(np.hstack((points2[matches23[:, 0], 0:2], points3[matches23[:, 1], 0:2]))), 0.5)
H12 = RANSAC(np.matrix(np.hstack((points1[matches12[:, 0], 0:2], points2[matches12[:, 1], 0:2]))), 0.5)
H01 = RANSAC(np.matrix(np.hstack((points0[matches01[:, 0], 0:2], points1[matches01[:, 1], 0:2]))), 0.5)

H43 = RANSAC(np.matrix(np.hstack((points4[matches43[:, 0], 0:2], points3[matches43[:, 1], 0:2]))), 0.5)
H54 = RANSAC(np.matrix(np.hstack((points5[matches54[:, 0], 0:2], points4[matches54[:, 1], 0:2]))), 0.5)

drawPoints(img0,points0)
drawPoints(img1,points1)
drawPoints(img2,points2)
drawPoints(img3,points3)
drawPoints(img4,points4)
drawPoints(img5,points5)

transformed_image2, offset_2 = appliqueTransformation(img2, H23[0])

transformed_image1, offset_1a = appliqueTransformation(img1, H12[0])
transformed_image1, offset_1b = appliqueTransformation(transformed_image1, H23[0])
offset_1 = np.add(offset_1a, offset_1b)

transformed_image0, offset_0a = appliqueTransformation(img0, H01[0])
transformed_image0, offset_0b = appliqueTransformation(transformed_image0, H12[0])
transformed_image0, offset_0c = appliqueTransformation(transformed_image0, H23[0])
offset_0 = np.add(np.add(offset_0a, offset_0b), offset_0c)

transformed_image4, offset_4 = appliqueTransformation(img4, H43[0])

transformed_image5, offset_5a = appliqueTransformation(img5, H54[0])
transformed_image5, offset_5b = appliqueTransformation(transformed_image5, H43[0])
offset_5 = np.add(offset_5a, offset_5b)

max_y_0 = max(abs(offset_0a[3]), abs(offset_0b[3]), abs(offset_0c[3]))
max_y_1 = max(abs(offset_1a[3]), abs(offset_1b[3]))
max_y_5 = max(abs(offset_5a[3]), abs(offset_5b[3]))

max_y = max(abs(offset_0[1]), abs(offset_1[1]), abs(offset_2[1]), abs(offset_4[1]), abs(offset_5[1])) + max(
    max_y_0, max_y_1, abs(offset_2[3]), abs(offset_4[3]), max_y_5)
max_x = max(abs(offset_0[0]), abs(offset_1[0]), abs(offset_2[0]), abs(offset_4[0]), abs(offset_5[0])) + max(
    abs(offset_0[2]), abs(offset_1[2]), abs(offset_2[2]), abs(offset_4[2]), abs(offset_5[2]))
new_width = max_x
new_height = max_y
final_image = np.zeros((new_height,
                        new_width, 3), dtype='uint8')

offset0 = [offset_0[0], offset_0[1]]
offset1 = [offset_1[0], offset_1[1]]
offset2 = [offset_2[0], offset_2[1]]
offset4 = [offset_4[0], offset_4[1]]
offset5 = [offset_5[0], offset_5[1]]

offset_global = [min(offset0[0], offset1[0], offset_2[0], offset_4[0], offset_5[0]),
                 min(offset0[1], offset1[1], offset_2[1], offset_4[1], offset_5[1])]

final_image[offset0[1] - offset_global[1]:offset0[1] - offset_global[1] + transformed_image0.shape[0],
offset0[0] - offset_global[0]:offset0[0] - offset_global[0] + transformed_image0.shape[1]] = transformed_image0

final_image[offset1[1] - offset_global[1]:offset1[1] - offset_global[1] + transformed_image1.shape[0],
offset1[0] - offset_global[0]:offset1[0] - offset_global[0] + transformed_image1.shape[1]] = transformed_image1

final_image[offset2[1] - offset_global[1]:offset2[1] - offset_global[1] + transformed_image2.shape[0],
offset2[0] - offset_global[0]:offset2[0] - offset_global[0] + transformed_image2.shape[1]] = transformed_image2

final_image[offset4[1] - offset_global[1]:offset4[1] - offset_global[1] + transformed_image4.shape[0],
offset4[0] - offset_global[0]:offset4[0] - offset_global[0] + transformed_image4.shape[1]] = transformed_image4

final_image[offset5[1] - offset_global[1]:offset5[1] - offset_global[1] + transformed_image5.shape[0],
offset5[0] - offset_global[0]:offset5[0] - offset_global[0] + transformed_image5.shape[1]] = transformed_image5

final_image[-offset_global[1]:-offset_global[1] + img3.shape[0],
-offset_global[0]:-offset_global[0] + img3.shape[1]] = img3

cv2.imwrite('golden_gate_points.jpg',final_image)


image0, image1, image2, image3 = transformRGBToGray(img0), transformRGBToGray(img1), transformRGBToGray(img2), transformRGBToGray(
    img3)
points0, points1, points2, points3 = harris_detector(image0, number=500), harris_detector(image1, number=500), harris_detector(
    image2, number=500), harris_detector(image3, number=500)


matches12 = matchTwoDescriptor(extraireDescripteur(image1, points1), extraireDescripteur(image2, points2))
matches01 = matchTwoDescriptor(extraireDescripteur(image0, points0), extraireDescripteur(image1, points1))

matches32 = matchTwoDescriptor(extraireDescripteur(image3, points3), extraireDescripteur(image2, points2))

H12 = RANSAC(np.matrix(np.hstack((points1[matches12[:, 0], 0:2], points2[matches12[:, 1], 0:2]))), 0.5)
H01 = RANSAC(np.matrix(np.hstack((points0[matches01[:, 0], 0:2], points1[matches01[:, 1], 0:2]))), 0.5)

H32 = RANSAC(np.matrix(np.hstack((points3[matches32[:, 0], 0:2], points2[matches32[:, 1], 0:2]))), 0.5)

# drawPoints(img0,points0)
# drawPoints(img1,points1)
# drawPoints(img2,points2)
# drawPoints(img3,points3)



transformed_image1, offset_1 = appliqueTransformation(img1, H12[0])

transformed_image0, offset_0a = appliqueTransformation(img0, H01[0])
transformed_image0, offset_0b = appliqueTransformation(transformed_image0, H12[0])
offset_0 = np.add(offset_0a, offset_0b)

transformed_image3, offset_3 = appliqueTransformation(img3, H32[0])



max_y_0 = max(abs(offset_0a[3]), abs(offset_0b[3]))


max_y = max(abs(offset_0[1]), abs(offset_1[1]), abs(offset_3[1])) + max(
    max_y_0,abs(offset_1[3]), abs(offset_3[3]))
max_x = max(abs(offset_0[0]), abs(offset_1[0]), abs(offset_3[0])) + max(
    abs(offset_0[2]), abs(offset_1[2]), abs(offset_3[2]))
new_width = max_x
new_height = max_y
final_image = np.zeros((new_height,
                        new_width, 3), dtype='uint8')

offset0 = [offset_0[0], offset_0[1]]
offset1 = [offset_1[0], offset_1[1]]
offset3 = [offset_3[0], offset_3[1]]

offset_global = [min(offset0[0], offset1[0], offset_3[0]),
                 min(offset0[1], offset1[1], offset_3[1])]


final_image[offset0[1] - offset_global[1]:offset0[1] - offset_global[1] + transformed_image0.shape[0],
offset0[0] - offset_global[0]:offset0[0] - offset_global[0] + transformed_image0.shape[1]] = transformed_image0

final_image[offset1[1] - offset_global[1]:offset1[1] - offset_global[1] + transformed_image1.shape[0],
offset1[0] - offset_global[0]:offset1[0] - offset_global[0] + transformed_image1.shape[1]] = transformed_image1


final_image[offset3[1] - offset_global[1]:offset3[1] - offset_global[1] + transformed_image3.shape[0],
offset3[0] - offset_global[0]:offset3[0] - offset_global[0] + transformed_image3.shape[1]] = transformed_image3



final_image[-offset_global[1]:-offset_global[1] + img2.shape[0],
-offset_global[0]:-offset_global[0] + img2.shape[1]] = img2

cv2.imwrite('pont.jpg',final_image)

image0, image1, image2, image3, image4, image5 = transformRGBToGray(img0), transformRGBToGray(img1), transformRGBToGray(
    img2), transformRGBToGray(
    img3), transformRGBToGray(img4), transformRGBToGray(img5)
points0, points1, points2, points3, points4, points5 = harris_detector(image0, number=500), harris_detector(image1,
                                                                                                            number=500), harris_detector(
    image2, number=500), harris_detector(image3, number=500), harris_detector(image4, number=500), harris_detector(
    image5, number=500)
points2_anms = pointSupressor(points2)
points3_anms = pointSupressor(points3)

matches04 = matchTwoDescriptor(extraireDescripteur(image0, points0), extraireDescripteur(image4, points4))
matches14 = matchTwoDescriptor(extraireDescripteur(image1, points1), extraireDescripteur(image4, points4))
matches24 = matchTwoDescriptor(extraireDescripteur(image2, points2), extraireDescripteur(image4, points4))

matches34 = matchTwoDescriptor(extraireDescripteur(image3, points3), extraireDescripteur(image4, points4))
matches54 = matchTwoDescriptor(extraireDescripteur(image5, points5), extraireDescripteur(image4, points4))

H04 = RANSAC(np.matrix(np.hstack((points0[matches04[:, 0], 0:2], points4[matches04[:, 1], 0:2]))), 0.5)
H14 = RANSAC(np.matrix(np.hstack((points1[matches14[:, 0], 0:2], points4[matches14[:, 1], 0:2]))), 0.5)
H24 = RANSAC(np.matrix(np.hstack((points2[matches24[:, 0], 0:2], points4[matches24[:, 1], 0:2]))), 0.5)

H34 = RANSAC(np.matrix(np.hstack((points3[matches34[:, 0], 0:2], points4[matches34[:, 1], 0:2]))), 0.5)
H54 = RANSAC(np.matrix(np.hstack((points5[matches54[:, 0], 0:2], points4[matches54[:, 1], 0:2]))), 0.5)

# drawPoints(img0,points0)
# drawPoints(img1,points1)
# drawPoints(img2,points2)
# drawPoints(img3,points3)
# drawPoints(img4,points4)
# drawPoints(img5,points5)

transformed_image0, offset_0 = appliqueTransformation(img0, H04[0])
transformed_image1, offset_1 = appliqueTransformation(img1, H14[0])
transformed_image2, offset_2 = appliqueTransformation(img2, H24[0])
transformed_image3, offset_3 = appliqueTransformation(img3, H34[0])
transformed_image5, offset_5 = appliqueTransformation(img5, H54[0])

max_y = max(abs(offset_0[1]), abs(offset_1[1]), abs(offset_2[1]), abs(offset_3[1]), abs(offset_5[1])) + max(
    abs(offset_0[3]), abs(offset_1[3]), abs(offset_2[3]), abs(offset_3[3]), abs(offset_5[3]))
max_x = max(abs(offset_0[0]), abs(offset_1[0]), abs(offset_2[0]), abs(offset_3[0]), abs(offset_5[0])) + max(
    abs(offset_0[2]), abs(offset_1[2]), abs(offset_2[2]), abs(offset_3[2]), abs(offset_5[2]))
new_width = max_x
new_height = max_y
final_image = np.zeros((new_height,
                        new_width, 3), dtype='uint8')

offset0 = [offset_0[0], offset_0[1]]
offset1 = [offset_1[0], offset_1[1]]
offset2 = [offset_2[0], offset_2[1]]
offset3 = [offset_3[0], offset_3[1]]
offset5 = [offset_5[0], offset_5[1]]

offset_global = [min(offset0[0], offset1[0], offset_2[0], offset_3[0], offset_5[0]),
                 min(offset0[1], offset1[1], offset_2[1], offset_3[1], offset_5[1])]

final_image[offset0[1] - offset_global[1]:offset0[1] - offset_global[1] + transformed_image0.shape[0],
offset0[0] - offset_global[0]:offset0[0] - offset_global[0] + transformed_image0.shape[1]] = transformed_image0

final_image[offset1[1] - offset_global[1]:offset1[1] - offset_global[1] + transformed_image1.shape[0],
offset1[0] - offset_global[0]:offset1[0] - offset_global[0] + transformed_image1.shape[1]] = transformed_image1

final_image[offset2[1] - offset_global[1]:offset2[1] - offset_global[1] + transformed_image2.shape[0],
offset2[0] - offset_global[0]:offset2[0] - offset_global[0] + transformed_image2.shape[1]] = transformed_image2

final_image[offset3[1] - offset_global[1]:offset3[1] - offset_global[1] + transformed_image3.shape[0],
offset3[0] - offset_global[0]:offset3[0] - offset_global[0] + transformed_image3.shape[1]] = transformed_image3

final_image[offset5[1] - offset_global[1]:offset5[1] - offset_global[1] + transformed_image5.shape[0],
offset5[0] - offset_global[0]:offset5[0] - offset_global[0] + transformed_image5.shape[1]] = transformed_image5

final_image[-offset_global[1]:-offset_global[1] + img4.shape[0],
-offset_global[0]:-offset_global[0] + img4.shape[1]] = img4

cv2.imwrite('plaines.jpg', final_image)

drawPoints(img0,points0)
drawPoints(img1,points1)
drawPoints(img2,points2)
drawPoints(img3,points3)
drawPoints(img4,points4)
drawPoints(img5,points5)

transformed_image0, offset_0 = appliqueTransformation(img0, H04[0])
transformed_image1, offset_1 = appliqueTransformation(img1, H14[0])
transformed_image2, offset_2 = appliqueTransformation(img2, H24[0])
transformed_image3, offset_3 = appliqueTransformation(img3, H34[0])
transformed_image5, offset_5 = appliqueTransformation(img5, H54[0])

max_y = max(abs(offset_0[1]), abs(offset_1[1]), abs(offset_2[1]), abs(offset_3[1]), abs(offset_5[1])) + max(
    abs(offset_0[3]), abs(offset_1[3]), abs(offset_2[3]), abs(offset_3[3]), abs(offset_5[3]))
max_x = max(abs(offset_0[0]), abs(offset_1[0]), abs(offset_2[0]), abs(offset_3[0]), abs(offset_5[0])) + max(
    abs(offset_0[2]), abs(offset_1[2]), abs(offset_2[2]), abs(offset_3[2]), abs(offset_5[2]))
new_width = max_x
new_height = max_y
final_image = np.zeros((new_height,
                        new_width, 3), dtype='uint8')

offset0 = [offset_0[0], offset_0[1]]
offset1 = [offset_1[0], offset_1[1]]
offset2 = [offset_2[0], offset_2[1]]
offset3 = [offset_3[0], offset_3[1]]
offset5 = [offset_5[0], offset_5[1]]

offset_global = [min(offset0[0], offset1[0], offset_2[0], offset_3[0], offset_5[0]),
                 min(offset0[1], offset1[1], offset_2[1], offset_3[1], offset_5[1])]

final_image[offset0[1] - offset_global[1]:offset0[1] - offset_global[1] + transformed_image0.shape[0],
offset0[0] - offset_global[0]:offset0[0] - offset_global[0] + transformed_image0.shape[1]] = transformed_image0

final_image[offset1[1] - offset_global[1]:offset1[1] - offset_global[1] + transformed_image1.shape[0],
offset1[0] - offset_global[0]:offset1[0] - offset_global[0] + transformed_image1.shape[1]] = transformed_image1

final_image[offset2[1] - offset_global[1]:offset2[1] - offset_global[1] + transformed_image2.shape[0],
offset2[0] - offset_global[0]:offset2[0] - offset_global[0] + transformed_image2.shape[1]] = transformed_image2

final_image[offset3[1] - offset_global[1]:offset3[1] - offset_global[1] + transformed_image3.shape[0],
offset3[0] - offset_global[0]:offset3[0] - offset_global[0] + transformed_image3.shape[1]] = transformed_image3

final_image[offset5[1] - offset_global[1]:offset5[1] - offset_global[1] + transformed_image5.shape[0],
offset5[0] - offset_global[0]:offset5[0] - offset_global[0] + transformed_image5.shape[1]] = transformed_image5

final_image[-offset_global[1]:-offset_global[1] + img4.shape[0],
-offset_global[0]:-offset_global[0] + img4.shape[1]] = img4

cv2.imwrite('plaines_points.jpg', final_image)





image0, image1, image2 = transformRGBToGray(img0), transformRGBToGray(img1), transformRGBToGray(img2)
points0, points1, points2 = harris_detector(image0, number=500), harris_detector(image1, number=500), harris_detector(
    image2, number=500)

matches01 = matchTwoDescriptor(extraireDescripteur(image0, points0), extraireDescripteur(image1, points1))
matches21 = matchTwoDescriptor(extraireDescripteur(image2, points2), extraireDescripteur(image1, points1))
print(matches21)

H01 = RANSAC(np.matrix(np.hstack((points0[matches01[:, 0], 0:2], points1[matches01[:, 1], 0:2]))), 0.5)

H21 = RANSAC(np.matrix(np.hstack((points2[matches21[:, 0], 0:2], points1[matches21[:, 1], 0:2]))), 0.5)
print(H21[0])

# drawPoints(img0, points0)
# drawPoints(img1, points1)
# drawPoints(img2, points2)

transformed_image0, offset_0 = appliqueTransformation(img0, H01[0])


transformed_image2, offset_2 = appliqueTransformation(img2, H21[0])
print(transformed_image2.shape)


max_y = max(abs((offset_0[1])), abs((offset_2[1]))) + max(abs((offset_0[3])), abs((offset_2[3])))
max_x = max(abs((offset_0[0])), abs((offset_2[0]))) + max(abs((offset_0[2])), abs((offset_2[2])))
new_width = max_x
new_height = max_y
final_image = np.zeros((new_height,
                        new_width, 3), dtype='uint8')

offset0 = [offset_0[0], offset_0[1]]
offset2 = [offset_2[0], offset_2[1]]

offset_global = [min(offset0[0], offset2[0]), min(offset0[1], offset2[1])]

final_image[offset0[1] - offset_global[1]:offset0[1] - offset_global[1] + transformed_image0.shape[0],
offset0[0] - offset_global[0]:offset0[0] - offset_global[0] + transformed_image0.shape[1]] = transformed_image0



final_image[offset2[1] - offset_global[1]:offset2[1] - offset_global[1] + transformed_image2.shape[0],
offset2[0] - offset_global[0]:offset2[0] - offset_global[0] + transformed_image2.shape[1]] = transformed_image2


final_image[-offset_global[1]:-offset_global[1] + img1.shape[0],
-offset_global[0]:-offset_global[0] + img1.shape[1]] = img1

cv2.imwrite('plt.jpg', final_image)
