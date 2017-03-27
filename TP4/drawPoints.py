import cv2
def drawPoints(im, arr):
    for elem in arr:
        tuple = (int(elem[0]), int(elem[1]))
        cv2.circle(im, tuple, 5, (58, 255, 78), thickness=2)