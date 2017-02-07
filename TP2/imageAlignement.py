import numpy as np
import scipy.misc as misc
import matplotlib.pyplot as plt


def norm_image(imageArray):
    return (imageArray.astype(np.float64) / 255.0)


def translate_image(img, t, axis):
    pad_width = [(0, 0), (0, 0)]
    if t > 0:
        pad_width[axis] = (t, 0)
    else:
        pad_width[axis] = (0, -t)
    pad_width = tuple(pad_width)
    return np.pad(img, pad_width, mode='constant')


def align_images(img1, img2):
    h1, w1 = img1.shape
    h2, w2 = img2.shape

    plt.imshow(img1, cmap='gray')
    x1, y1 = tuple(zip(*plt.ginput(2)))
    cx1, cy1 = np.mean(x1), np.mean(y1)
    plt.imshow(img2, cmap='gray')
    x2, y2 = tuple(zip(*plt.ginput(2)))
    cx2, cy2 = np.mean(x2), np.mean(y2)

    plt.close()
    tx = int(np.round((w1 / 2 - cx1) * 2))
    img1 = translate_image(img1, tx, axis=1)
    ty = int(np.round((h1 / 2 - cy1) * 2))
    img1 = translate_image(img1, ty, axis=0)
    tx = int(np.round((w2 / 2 - cx2) * 2))
    img2 = translate_image(img2, tx, axis=1)
    ty = int(np.round((h2 / 2 - cy2) * 2))
    img2 = translate_image(img2, ty, axis=0)

    len1 = np.sqrt((y1[1] - y1[0]) ** 2 + (x1[1] - x1[0]) ** 2)
    len2 = np.sqrt((y2[1] - y2[0]) ** 2 + (x2[1] - x2[0]) ** 2)

    dscale = len2 / len1

    if dscale < 1:
        img1 = misc.imresize(img1, dscale, 'bilinear')
    else:
        img2 = misc.imresize(img2, 1 / dscale, 'bilinear')

    theta1 = np.arctan2(-(y1[1] - y1[0]), x1[1] - x1[0])
    theta2 = np.arctan2(-(y2[1] - y2[0]), x2[1] - x2[0])
    dtheta = theta2 - theta1
    img1 = misc.imrotate(img1, dtheta * 180 / np.pi, 'bilinear')
    img1 = norm_image(img1)  # imrotate semble remettre l'image en [0-255]

    h1, w1 = img1.shape
    h2, w2 = img2.shape

    minw = min(w1, w2)
    brd = (max(w1, w2) - minw) / 2

    if minw == w1:
        img2 = img2[:, int(np.ceil(brd)):-int(np.floor(brd))]
    else:
        img1 = img1[:, int(np.ceil(brd)):-int(np.floor(brd))]

    minh = min(h1, h2)
    brd = (max(h1, h2) - minh) / 2

    if minh == h1:
        img2 = img2[int(np.ceil(brd)):-int(np.floor(brd)), :]
    else:
        img1 = img1[int(np.ceil(brd)):-int(np.floor(brd)), :]

    return img1, img2
