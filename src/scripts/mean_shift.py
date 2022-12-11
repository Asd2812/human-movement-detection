import numpy as np
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='Meanshift algorithm')

parser.add_argument('image', type=str, help='path to image file')
args = parser.parse_args()
cap = cv.VideoCapture(args.image)

# First frame of video
ret, frame = cap.read()

# Hardcoded initial window of tracking
x, y, w, h = 800, 500, 100, 100
track_window = (x, y, w, h)

# Set up ROI for tracking
roi = frame[y:y+h, x:x+w]
hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

# Setup termination criteria: 10 iteration or move by at least 1 pt
term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

while(1):
    ret, frame = cap.read()
    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # apply meanshift to get the new location
        ret, track_window = cv.meanShift(dst, track_window, term_crit)

        # Draw it on image
        x,y,w,h = track_window
        track = cv.rectangle(frame, (x, y), (x+w, y+h), 255, 2)
        cv.imshow('Mia Tracking', track)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break
