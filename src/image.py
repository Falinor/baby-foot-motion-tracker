import numpy as np
import argparse
import cv2
import sys


MIN_DIST = 50
PARAM_1 = 40
PARAM_2 = 20
MIN_RADIUS = 7
MAX_RADIUS = 10

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "Path to the image")
args = vars(ap.parse_args())

# load the image, clone it for output, and then convert it to grayscale
image = cv2.imread(args["image"])
output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect circles in the image
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, MIN_DIST, param1=PARAM_1, param2=PARAM_2, minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS)

try:
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            # show the output image
            cv2.imshow("output", np.hstack([image, output]))
            cv2.waitKey(0)

except KeyboardInterrupt:
    sys.exit()