import cv2
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread('box.png', cv.IMREAD_GRAYSCALE) # queryImage
img2 = cv.imread('box_in_scene.png', cv.IMREAD_GRAYSCALE) # trainImage

# Initiate SIFT detector
sift = cv.SIFT_create()

# fin the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)

# Step 3: Apply the 8-point algorithm and find the fundamental matrix F
if len(good) >= 8:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])
    F, mask = cv.findFundamentalMat(src_pts,dst_pts,cv.FM_8POINT)

    # Step 4: Draw epipolar lines in img2
    lines = cv.computeCorrespondEpilines(dst_pts.reshape(-1, 2), 2, F)
    lines = lines.reshape(-1, 3)

    img2_with_lines = cv.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    for r, pt1, pt2 in zip(lines, src_pts, dst_pts):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [img2.shape[1], -(r[2]+r[0]*img2.shape[1])/r[1]])
        img2_with_lines = cv.line(img2_with_lines, (x0, y0), (x1, y1), color, 1)
    cv.imshow('Epipolar Lines', img2_with_lines)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# cv.drawMatchesKnn expects list of lists as matches.
# img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# plt.imshow(img3),plt.show()

