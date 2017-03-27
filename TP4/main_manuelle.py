import cv2
import numpy as np
from numpy.linalg import inv
from math import floor
from TP4.appliqueTransformation import appliqueTransformation
from TP4.gatherPoints import gatherPoints
from TP4.calculerHomographie import getPointsFromFile, calculerHomographie
from TP4.drawPoints import drawPoints

# gatherPoints('images/1- PartieManuelle/Serie1/IMG_2415.JPG',"pts1_12.txt")
# gatherPoints('images/1- PartieManuelle/Serie1/IMG_2416.JPG',"pts2_12.txt")
# gatherPoints('images/1- PartieManuelle/Serie1/IMG_2416.JPG',"pts2_32.txt")
# gatherPoints('images/1- PartieManuelle/Serie1/IMG_2417.JPG',"pts3_32.txt")
#
# gatherPoints('images/1- PartieManuelle/Serie2/IMG_2425.JPG',"2_pts1_12.txt")
# gatherPoints('images/1- PartieManuelle/Serie2/IMG_2426.JPG',"2_pts2_12.txt")
# gatherPoints('images/1- PartieManuelle/Serie2/IMG_2426.JPG',"2_pts2_32.txt")
# gatherPoints('images/1- PartieManuelle/Serie2/IMG_2427.JPG',"2_pts3_32.txt")
#
# gatherPoints('images/1- PartieManuelle/Serie3/IMG_2409.JPG',"3_pts1_12.txt")
# gatherPoints('images/1- PartieManuelle/Serie3/IMG_2410.JPG',"3_pts2_12.txt")
# gatherPoints('images/1- PartieManuelle/Serie3/IMG_2410.JPG',"3_pts2_32.txt")
# gatherPoints('images/1- PartieManuelle/Serie3/IMG_2411.JPG',"3_pts3_32.txt")

# #
# im1_pts_1 = getPointsFromFile("pts_serie1/pts1_12.txt")
# im2_pts_1 = getPointsFromFile("pts_serie1/pts2_12.txt")
# im2_pts_2 = getPointsFromFile("pts_serie1/pts2_32.txt")
# im3_pts_2 = getPointsFromFile("pts_serie1/pts3_32.txt")

# im1_pts_1 = getPointsFromFile("2_pts1_12.txt")
# im2_pts_1 = getPointsFromFile("2_pts2_12.txt")
# im2_pts_2 = getPointsFromFile("2_pts2_32.txt")
# im3_pts_2 = getPointsFromFile("2_pts3_32.txt")

im1_pts_1 = getPointsFromFile("3_pts1_12.txt")
im2_pts_1 = getPointsFromFile("3_pts2_12.txt")
im2_pts_2 = getPointsFromFile("3_pts2_32.txt")
im3_pts_2 = getPointsFromFile("3_pts3_32.txt")

H12 = calculerHomographie(im1_pts_1, im2_pts_1)
H32 = calculerHomographie(im3_pts_2, im2_pts_2)
#
# img1 = cv2.imread('images/1- PartieManuelle/Serie1/IMG_2415.JPG', 1)
# img2 = cv2.imread('images/1- PartieManuelle/Serie1/IMG_2416.JPG', 1)
# img3 = cv2.imread('images/1- PartieManuelle/Serie1/IMG_2417.JPG', 1)

# img1 = cv2.imread('images/1- PartieManuelle/Serie2/IMG_2425.JPG', 1)
# img2 = cv2.imread('images/1- PartieManuelle/Serie2/IMG_2426.JPG', 1)
# img3 = cv2.imread('images/1- PartieManuelle/Serie2/IMG_2427.JPG', 1)


img1 = cv2.imread('images/1- PartieManuelle/Serie3/IMG_2409.JPG', 1)
img2 = cv2.imread('images/1- PartieManuelle/Serie3/IMG_2410.JPG', 1)
img3 = cv2.imread('images/1- PartieManuelle/Serie3/IMG_2411.JPG', 1)

# points = [(612, 267), (713, 271), (807, 315), (468, 393)]
# for pt in points:
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     cv2.putText(current_image, str(points.index(pt) + 1), pt, font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
#     cv2.circle(current_image, pt, 5, (255, 0, 0), thickness=1)
#
# cv2.imshow('img', current_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


transformed_image1, offset_1 = appliqueTransformation(img1, H12)
transformed_image3, offset_3 = appliqueTransformation(img3, H32)

# drawPoints(img1, im1_pts_1)
# drawPoints(img2, im2_pts_1)
# drawPoints(img2, im2_pts_2)
# drawPoints(img3, im3_pts_2)

top_right_img_2_y = offset_1[1] + img2.shape[0]
top_right_img_2_x = offset_1[0] + img2.shape[1]
bottom_right_img_3_y = top_right_img_2_y + offset_3[3]
bottom_right_img_3_x = top_right_img_2_x + offset_3[2]

max_y = max(abs((offset_1[1])), abs((offset_3[1]))) + max(abs((offset_1[3])), abs((offset_3[3])))
max_x = max(abs((offset_1[0])), abs((offset_3[0]))) + max(abs((offset_1[2])), abs((offset_3[2])))
new_width = max_x
new_height = max_y
final_image = np.zeros((new_height,
                        new_width, 3), dtype='uint8')

offset1 = [offset_1[0], offset_1[1]]
offset3 = [offset_3[0], offset_3[1]]

offset_global = [min(offset1[0], offset3[0]), min(offset1[1], offset3[1])]
print("Offset global", offset_global)
print("offset image 1", offset1)
print("offset image 3", offset3)

final_image[offset1[1] - offset_global[1]:offset1[1] - offset_global[1] + transformed_image1.shape[0],
offset1[0] - offset_global[0]:offset1[0] - offset_global[0] + transformed_image1.shape[1]] = transformed_image1

top_left_img_2_y = abs((offset_1[1]))
top_left_img_2_x = abs((offset_1[0]))

final_image[offset3[1] - offset_global[1]:offset3[1] - offset_global[1] + transformed_image3.shape[0],
offset3[0] - offset_global[0]:offset3[0] - offset_global[0] + transformed_image3.shape[1]] = transformed_image3

final_image[-offset_global[1]:-offset_global[1] + img2.shape[0],
-offset_global[0]:-offset_global[0] + img2.shape[1]] = img2

cv2.imshow('img', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('manuelle3.jpg', final_image)
