import cv2
from matplotlib import pyplot as plt
import numpy as np
import scipy.misc


img1 = cv2.imread('nfl_1.jpg')
img2 = cv2.imread('nfl_2.jpg')

# plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
kernel_sharpen_1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
output_1 = cv2.filter2D(img1, -1, kernel_sharpen_1)
output_2 = cv2.filter2D(img2, -1, kernel_sharpen_1)
scipy.misc.imsave("nfl_1_sharpened.png", cv2.cvtColor(output_1, cv2.COLOR_BGR2RGB))
scipy.misc.imsave("nfl_2_sharpened.png", cv2.cvtColor(output_2, cv2.COLOR_BGR2RGB))
