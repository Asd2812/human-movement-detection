import cv2 as cv
import numpy as np

# The video feed is read in as a VideoCapture object
cap = cv.VideoCapture("trimmed_arman_walking.mp4")
# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
ret, first_frame = cap.read()
# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
# Creates an image filled with zero intensities with the same dimensions as the frame
mask = np.zeros_like(first_frame, np.uint8)
# Sets image saturation to maximum
mask[..., 1] = 255

while(cap.isOpened()):
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    ret, frame = cap.read()
    # Opens a new window and displays the input frame
    cv.imshow("input", frame)
    frame = frame.astype("uint8")
    # Converts each frame to grayscale - we previously only converted the first frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Calculates dense optical flow by Farneback method
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    print("Flow:\n", type(flow))
    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    print("Magnitude:\n", type(magnitude))
    # Sets image hue according to the optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2
    # Sets image value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    print("Mask:\n", mask[1])
    # Converts HSV to RGB (BGR) color representation
    spec_mask = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
    spec_mask = cv.cvtColor(spec_mask, cv.COLOR_BGR2GRAY)
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
    print(spec_mask.shape)
    spec_mask = spec_mask.resize((1922, 1080))
    rgb = rgb.astype("uint8")
    cv.floodFill(rgb, spec_mask, (1000, 600), 255)
    # Opens a new window and displays the output frame
    cv.imshow("dense optical flow", rgb)
    # Update frame
    prev_gray = gray
    # Frames intervals of 1 ms
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()