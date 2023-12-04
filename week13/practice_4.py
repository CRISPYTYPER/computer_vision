import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

img_l = cv2.imread("./im6.png", cv2.IMREAD_GRAYSCALE)
img_r = cv2.imread("./im2.png", cv2.IMREAD_GRAYSCALE)

# 오른쪽 이미지가 항상 오른쪽에 있음
print(img_l.shape)
print(img_r.shape)

(height, width) = img_l.shape

window_size = 11
window_size_half = window_size//2



disparity_arr = np.zeros((height, width))

for i in tqdm(range(window_size_half, height - window_size_half)): # target h
    for j in range(window_size_half, width - window_size_half): # target w
        target_window = img_l[i - window_size_half:i + window_size_half + 1,
                        j - window_size_half: j + window_size_half + 1]
        compare_window = img_r[i - window_size_half:i + window_size_half + 1,
                         j - window_size_half:j + window_size_half + 1]
        min_cost = np.sum(np.square(compare_window - target_window))
        # print(min_cost)
        horizontal_diff = 0
        for k in range(j, width - window_size_half):
            # 비교대상(left)
            target_window = img_l[i-window_size_half:i+window_size_half+1, j-window_size_half: j+window_size_half+1]
            compare_window = img_r[i-window_size_half:i+window_size_half+1, k - window_size_half:k + window_size_half + 1]
            absolute_difference = np.sum(np.square(compare_window - target_window))
            # print(absolute_difference)
            if(min_cost > absolute_difference):
                min_cost = absolute_difference
                horizontal_diff = k - j
            disparity_arr[i][j] = horizontal_diff

disparity_arr = disparity_arr / np.max(disparity_arr)
print(disparity_arr)
plt.imshow(disparity_arr * 255)
