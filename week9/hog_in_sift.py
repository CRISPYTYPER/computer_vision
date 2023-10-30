import numpy as np
from PIL import Image
import scipy
import cv2 as cv

img = cv.imread("scale1.png")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,3,0.04)

print(dst.shape, dst[100][100], dst.max())

img[dst>0.01*dst.max()] = [0,0,255]

cv.imshow('dst', img)
key = cv.waitKey(0)
if key == 27 or key == 113:  # Check for 'ESC' (27) or 'q' (113) key press
    cv.destroyAllWindows()



