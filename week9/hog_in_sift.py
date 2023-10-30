import math

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

# coordinates = np.where(dst > 0.01 * dst.max())
coordinates = np.where(dst > 0.1 * dst.max())

print(coordinates[0][250], coordinates[1][250])
point_x = coordinates[0][250]
point_y = coordinates[1][250]
window_size = 23
window = gray[point_y - window_size // 2:point_y + window_size // 2 + 1, point_x - window_size // 2:point_x + window_size // 2 + 1]

filter_x = np.zeros((3, 3))
filter_x[:, 0] = [1, 2, 1]
filter_x[:, 2] = [-1, -2, -1]


filter_y = np.zeros((3, 3))
filter_y[0, :] = [1, 2, 1]
filter_y[2, :] = [-1, -2, -1]

#arrays to store gradients Ix and Iy

Ix = scipy.signal.convolve2d(window, filter_x, boundary='fill', fillvalue=0)
Iy = scipy.signal.convolve2d(window, filter_y, boundary='fill', fillvalue=0)

gradient = [Ix, Iy]
direction = np.arctan2(Iy,Ix)

hog_list = [0] * 36
for row in direction:
    for element in row:
        element_degrees = math.degrees(element)
        if element_degrees < 0:
            element_degrees += 360
        hog_list[int(element_degrees / 10)] += 1
max_index = np.argmax(hog_list)
print(hog_list)
print(f"max index: {max_index}")




cv.imshow('dst', img)
if cv.waitKey(0) & 0xff == 27:  # Check for 'ESC' (27) or 'q' (113) key press
    cv.destroyAllWindows()



