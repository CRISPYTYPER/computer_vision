import math

import numpy as np
from PIL import Image
import scipy
import cv2 as cv
import random

img = cv.imread("scale1.png")
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,3,0.04)

idx = dst>0.01*dst.max()

(img_h, img_w) = dst.shape 
corner_thr = 0.01*dst.max()

features = []
for i in range(0, img_h):
    for j in range(0, img_w):
        if dst[i][j] > corner_thr:#==corner!
            # img[i][j]=[0,0,255]
            features.append((j,i))

#step 1.
(feat_x, feat_y) = random.choice(features) # a single corner point (x, y), boundary condition chk!
print((feat_x, feat_y))

#step 2.
patch_size = 23
patch_size_half = patch_size//2 
W = gray[feat_y-patch_size_half:feat_y+patch_size_half+1, feat_x-patch_size_half:feat_x+patch_size_half+1]/255.0 #23x23 patch, brightness range=(0,1)


#step 3
# df_dx = conv(W, s_x)
# dw_dy = conv(W, s_y)

filter_x = np.zeros((3, 3))
filter_x[:, 0] = [1, 2, 1]
filter_x[:, 2] = [-1, -2, -1]

filter_y = np.zeros((3, 3))
filter_y[0, :] = [1, 2, 1]
filter_y[2, :] = [-1, -2, -1]
df_dx = scipy.signal.convolve2d(W, filter_x, boundary='fill', fillvalue=0)
df_dy = scipy.signal.convolve2d(W, filter_y, boundary='fill', fillvalue=0)

gradient = [df_dx, df_dy]
direction = np.arctan2(df_dy,df_dx)

#step 4.
angle_histogram = np.ones(36)#36bins
# for i in range(patch_size):
#     for j in range(patch_size):
#         angle = XX
#         angle_histogram[angle] +=1

for row in direction:
    for element in row:
        element_degrees = math.degrees(element)
        if element_degrees < 0:
            element_degrees += 360
        angle_histogram[int(element_degrees / 10)] += 1
max_index = np.argmax(angle_histogram)
print(angle_histogram)
print(f"max index: {max_index}")

#step 5.
dominant_angle = max(angle_histogram)#degree (not radian)
# print(dominant_angle)

#step 6. do rotation normalization
rot_matrix = cv.getRotationMatrix2D((patch_size/2, patch_size/2), -dominant_angle, 1)
rot_W = cv.warpAffine(W, rot_matrix, (patch_size, patch_size))

#step 7. extract 16x16 window
w = rot_W[patch_size_half-8 : patch_size_half+8, patch_size_half-8 : patch_size_half+8]#16x16 window w
# print(w.shape)

#step 8. divide w into 4x4 cells
cell = [[] for i in range(16)]
for i in range(4):
    for j in range(4):
        cell[i*4+j] = w[4*i:4*(i+1), 4*j:4*(j+1)]
# cell[0] = w[0:4, 0:4]
# cell[1] = w[4:8, 0:4]
# cell[2] = w[8:12, 0:4]
# cell[3] = w[12:16, 0:4]
# ...

#step 9. build hog for each cell
hog = [[0 for j in range(8)] for i in range(16)]
for k in range(16): # 각 cell 순회용
    for i in range(4): # 세로 진행용
        for j in range(4): # 가로 진행용
            for q in range(8): # Hog 제작용
                if(cell[k][i][j]>=q*0.125 and cell[k][i][j]<(q+1)*0.125):
                    hog[k][q] += 1
#hog[0] = hog from cell[0]
#hog[1] = hog from cell[1]

#step 10. construct 128-dim dscr
sift_dscr = []
for n in range(16):
    sift_dscr = sift_dscr + hog[n]

print(sift_dscr)