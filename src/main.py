from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import time


MIN_DIST = 100
PARAM_1 = 50
PARAM_2 = 20
MIN_RADIUS = 22
MAX_RADIUS = 28

INTERNAL_CAMERA = 0
EXTERNAL_CAMERA = 1

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())

if not args.get("video", False):
    camera = cv2.VideoCapture(EXTERNAL_CAMERA)
else:
    camera = cv2.VideoCapture(args["video"])

while True:
    (grabbed, frame) = camera.read()

    if args.get("video") and not grabbed:
        break

    frame = imutils.resize(frame, width=1600)
    frame = frame[260:870, 130:1300]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.erode(frame, None, iterations=2)
    frame = cv2.dilate(frame, None, iterations=2)

    # detect circles in the image
    circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, 1.2, MIN_DIST, param1=PARAM_1, param2=PARAM_2, minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS)

    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    cv2.imshow("Frame", frame)
    time.sleep(1/15)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
