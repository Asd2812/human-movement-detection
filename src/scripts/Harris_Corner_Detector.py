# Import Statements
import numpy as np
import cv2 as cv

filename = 'images/Arman_Static.jpeg'
img = cv.imread(filename)
#grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

gray = np.float32(gray)
#gray = np.float32(img)
dst = cv.cornerHarris(gray, 2, 3, 0.04)

# result is dilated for marking the corners, not important
dst = cv.dilate(dst, None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst > 0.01 * dst.max()] = [0, 0, 255]
result = cv.imwrite('images/Arman_Static_Harris.png', img)

if result == True:
  print("File saved successfully")
else:
  print("Error in saving file")


