# import the necessary packages
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
import matlab.engine
import numpy as np
from PIL import Image as im2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
args = vars(ap.parse_args())

# Read from video file
vs = cv2.VideoCapture(args["video"])

# Initialize first frame
firstFrame = None

# Loop over frames
eng = matlab.engine.start_matlab()
eng.addpath('~/Downloads/AOFSkeletons')
k = 0
while True:
    # Get current frame
    frame = vs.read()
    frame = frame if args.get("video", None) is None else frame[1]

    # Check for end of the video
    if frame is None:
        break

    # Resize, grayscale, and blur
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # If no first frame, initialize with basic gray image
    if firstFrame is None:
        firstFrame = gray
        continue

    # compute the absolute difference between the current frame and first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    # Dilate thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    # Get countours
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    db = matlab.uint8(thresh)
    skel = (eng.generate_skeletons(db, nargout=1))
    print(type(skel))
    fm = im2.fromarray(np.asarray(skel))

    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)
    cv2.imshow("Skeley Bois", np.array(skel))
    key = cv2.waitKey(1) & 0xFF
    if k == 50:
        cv2.imsave('thresh_im.jpg', thresh)
        cv2.imsave('fd_im.jpg', frameDelta)
        cv2.imsave('skel_im.jpg', np.array(skel))
    # If the `e` key is pressed, break
    if key == ord("e"):
        break

# Close
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()
eng.quit()
