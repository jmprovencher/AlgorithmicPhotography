# Usage : Press ESC when you are done taking points, press SPACE to remove the last point clicked.

import argparse
import cv2
import numpy as np
import imutils


refPt = []

def click_gather_point(event, x, y, flags, param):
    global refPt
    if event == cv2.EVENT_LBUTTONDOWN:
        print(len(refPt))
        point = (x, y)
        refPt.append(point)
        font = cv2.FONT_HERSHEY_SIMPLEX



def gatherPoints(img_filename, save_file_path):
    global refPt
    refPt = []
    cv2.namedWindow("image")

    # load the image, clone it, and setup the mouse callback function

    image = cv2.imread(img_filename)
    current_image = image.copy()
    cv2.setMouseCallback("image", click_gather_point)
    while len(refPt) <= 3:
        cv2.imshow("image", current_image)
        key = cv2.waitKey(1) & 0xFF
        if (key == 32 and len(refPt) > 0):
            refPt.pop()
            current_image = image.copy()
            for pt in refPt:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(current_image, str(refPt.index(pt) + 1), pt, font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(current_image, pt, 5, (255, 0, 0), thickness=1)
        elif (key == 27):
            break;

    filename = save_file_path
    file = open(filename, "w")
    [file.write(str(pt[0]) + "," + str(pt[1]) + "\n") for pt in refPt]
    file.close()


    # close all open windows
    cv2.destroyAllWindows()


