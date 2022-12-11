# import the necessary packages
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
args = vars(ap.parse_args())

# Read from video file
vs = cv2.VideoCapture(args["video"])

# Initialize first frame
firstFrame = None

# Loop over frames
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

    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)
    key = cv2.waitKey(1) & 0xFF
    # If the `e` key is pressed, break
    if key == ord("e"):
        break

# Close
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()
