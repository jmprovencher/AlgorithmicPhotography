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
        cv2.putText(current_image, str(len(refPt)), point, font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(current_image, point, 5, (255, 0, 0), thickness=1)


cv2.namedWindow("image")
cv2.setMouseCallback("image", click_gather_point)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image, clone it, and setup the mouse callback function

image = imutils.resize(cv2.imread(args["image"]), width=800)
current_image = image.copy()
while True:
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

file = open("output.txt", "w")
[file.write(str(pt) + "\n") for pt in refPt]
file.close()

# close all open windows
cv2.destroyAllWindows()