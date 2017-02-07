import cv2
from matplotlib import pyplot as plt
from TP2.imageAlignement import align_images, norm_image
from TP2.hybridImage import hybridImage
albert = cv2.imread('Albert_Einstein.png', cv2.IMREAD_GRAYSCALE)
marilyn = cv2.imread('Marilyn_Monroe.png', cv2.IMREAD_GRAYSCALE)

# plt.imshow(albert, cmap='gray')
# plt.show()
# plt.imshow(marilyn, cmap='gray')
# plt.show()

arbitrary_value_1 = 25
arbitrary_value_2 = 10
cutoff_low = arbitrary_value_1
cutoff_high = arbitrary_value_2


albert, marilyn = align_images(albert, marilyn)
hybrid_img = hybridImage(albert, marilyn, cutoff_low, cutoff_high)
plt.imshow(hybrid_img, cmap="gray")
plt.show()


