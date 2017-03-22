import cv2
import numpy as np
from numpy.linalg import inv
from math import floor, ceil
from TP4.appliqueTransformation import appliqueTransformation



img = cv2.imread('images/0- Rechauffement/JFLalonde.png', 1)
H = np.array([[-0.0814, 0.7402, 34.3412], [1.0209, 0.1534, 60.3258], [0.0005, 0, 1.0000]])

transformed_image = appliqueTransformation(img, H)
cv2.imshow('img', transformed_image)
cv2.imwrite('JFLalonde_transformed.png',transformed_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

