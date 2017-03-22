import os
import glob
import cv2
import numpy as np
from TP3.utils import delaunay, getPointsFromFile


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
    for j in range(0, len(img1_pts)):
        x = (1 - warp_frac) * img1_pts[j][0] + warp_frac * img2_pts[j][0]
        y = (1 - warp_frac) * img1_pts[j][1] + warp_frac * img2_pts[j][1]
        points.append((x, y))

    for element in tri.simplices:
        x = element[0]
        y = element[1]
        z = element[2]

        x = int(x)
        y = int(y)
        z = int(z)

        t1 = [img1_pts[x], img1_pts[y], img1_pts[z]]
        t2 = [img2_pts[x], img2_pts[y], img2_pts[z]]
        t = [points[x], points[y], points[z]]

        morphTriangle(img1, img2, imgMorph, t1, t2, t, dissolve_frac)

    return imgMorph.astype('u1')


if __name__ == '__main__':

    images_points = []
    images_points_student = []
    images_points_utrecht = []
    images_points_utrecht_male = []
    images_points_utrecht_female = []
    for f in glob.glob(os.path.join('points_created', "*.txt")):
        line = open(f).readlines()
        images_points.append(line)

    for f in glob.glob(os.path.join('points', "*.txt")):
        line = getPointsFromFile(f)
        images_points_student.append(line)

    for f in glob.glob(os.path.join('points_created_utrecht', "*.txt")):
        filename, file_extension = os.path.splitext(f)
        if filename[-1] != 's':
            line = getPointsFromFile(f)
            images_points_utrecht.append(line)

    for f in glob.glob(os.path.join('points_created_utrecht_male', "*.txt")):
        filename, file_extension = os.path.splitext(f)
        if filename[-1] != 's':
            line = getPointsFromFile(f)
            images_points_utrecht_male.append(line)

    for f in glob.glob(os.path.join('points_created_utrecht_female', "*.txt")):
        filename, file_extension = os.path.splitext(f)
        if filename[-1] != 's':
            line = getPointsFromFile(f)
            images_points_utrecht_female.append(line)

    mean_points = []
    mean_points_student = []
    mean_points_utrecht = []
    mean_points_utrecht_male = []
    mean_points_utrecht_female = []

    number_of_images = len(images_points)
    number_of_points = len(images_points[0])

    number_of_images_student = len(images_points_student)
    number_of_points_student = len(images_points_student[0])

    number_of_images_utrecht = len(images_points_utrecht)
    number_of_points_utrecht = len(images_points_utrecht[0])

    number_of_images_utrecht_male = len(images_points_utrecht_male)
    number_of_points_utrecht_male = len(images_points_utrecht_male[0])

    number_of_images_utrecht_female = len(images_points_utrecht_female)
    number_of_points_utrecht_female = len(images_points_utrecht_female[0])

    # Moyenne pondérée des coordonnées de points
    for i in range(74):  # chaque
        mean_x = int(sum([int(x[i][0:3]) for x in images_points]) / number_of_images)
        mean_y = int(sum([int(y[i][4:7]) for y in images_points]) / number_of_images)
        mean_points.append((mean_x, mean_y))

    for i in range(number_of_points_student):  # chaque
        mean_x = int(sum([int(x[i][0]) for x in images_points_student]) / number_of_images_student)
        mean_y = int(sum([int(y[i][1]) for y in images_points_student]) / number_of_images_student)
        mean_points_student.append((mean_x, mean_y))

    for i in range(number_of_points_utrecht):  # chaque
        mean_x = int(sum([int(x[i][0]) for x in images_points_utrecht]) / number_of_images_utrecht)
        mean_y = int(sum([int(y[i][1]) for y in images_points_utrecht]) / number_of_images_utrecht)
        mean_points_utrecht.append((mean_x, mean_y))

    for i in range(number_of_points_utrecht_male):  # chaque
        mean_x = int(sum([int(x[i][0]) for x in images_points_utrecht_male]) / number_of_images_utrecht_male)
        mean_y = int(sum([int(y[i][1]) for y in images_points_utrecht_male]) / number_of_images_utrecht_male)
        mean_points_utrecht_male.append((mean_x, mean_y))

    for i in range(number_of_points_utrecht_female):  # chaque
        mean_x = int(sum([int(x[i][0]) for x in images_points_utrecht_female]) / number_of_images_utrecht_female)
        mean_y = int(sum([int(y[i][1]) for y in images_points_utrecht_female]) / number_of_images_utrecht_female)
        mean_points_utrecht_female.append((mean_x, mean_y))

    list_of_morphed_images = []

    canvas = np.zeros((1200, 900, 3), np.uint8)

    for point in mean_points_utrecht_male:
        cv2.circle(canvas, (point[0], point[1]), 3, (0, 0, 255), -1)

    for f in glob.glob(os.path.join('utrecht', "*.jpg")):
        img1 = cv2.imread(f)
        filename, file_extension = os.path.splitext(f)
        if filename[-1] != 's':
            directory, image_name = os.path.split(filename)
            if image_name[0] != 'f':
                img2 = np.zeros((1200, 900, 3))
                points1 = getPointsFromFile("points_created_utrecht_male/" + image_name + ".txt")
                points2 = mean_points_utrecht
                tri_points = (points1 + points2) / 2
                imagesArray = []
                tri = delaunay(tri_points)

                list_of_morphed_images.append(morph(img1, img2, points1, points2, tri, 1, 0))

    N = len(list_of_morphed_images)

    # Create a numpy array of floats to store the average (assume RGB images)
    arr = np.zeros((1200, 900, 3))

    # Build up average pixel intensities, casting each image as an array of floats
    for im in list_of_morphed_images:
        arr = np.add(arr, im)

    final_image = arr / N
    cv2.imwrite('mean_visage_utrecht_male.png', final_image.astype('uint8'))

    img1 = cv2.imread('jm/jm.jpg')
    img2 = cv2.imread('mean_visage_utrecht_female.png')

    points1 = getPointsFromFile('jm/jm.txt')
    points2 = mean_points_utrecht_female
    tri_points = (points1 + points2) / 2
    tri = delaunay(tri_points)

    morphed_image = morph(img1, img2, points1, points2, tri, 1, 0)

    cv2.imwrite('jm_to_female_2.png', morphed_image.astype('uint8'))
