import cv2
import numpy as np

img0 = cv2.imread("sphere0.png")
img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
img1 = cv2.imread("sphere1.png")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
(width, height) = img0.shape
harris_response = cv2.cornerHarris(np.float32(img0),2,3,0.04)
corner_idx = harris_response > 0.01 * np.max(harris_response)

filter_x = np.zeros((3, 3))
filter_x[1, 1:3] = [1, -1]
filter_x = np.flip(filter_x, axis=None) # flip-flop


filter_y = np.zeros((3, 3))
filter_y[1:3, 1] = [1, -1]
filter_y= np.flip(filter_y, axis=None) # flip-flop

# Initialize arrays to store gradients Ix and Iy
Ix = np.zeros_like(img0)
Iy = np.zeros_like(img0)

for x in range(1, width-1):
    for y in range(1, height-1):
        tmp = img0[x - 1:x + 2, y - 1:y + 2] # 3x3 box
        conv_x = np.sum(filter_x * tmp)
        conv_y = np.sum(filter_y * tmp)
        Ix[x,y] = conv_x
        Iy[x,y] = conv_y

# Define the window size for local computation
window_size = 5

# Initialize arrays to store components
I_x_squared = np.zeros_like(img0)
I_y_squared = np.zeros_like(img0)
I_x_y = np.zeros_like(img0)


for 
# Compute components within the window
for x in range( + window_size // 2, width - window_size//2 - 1):
    for y in range(1 + window_size // 2, height - window_size//2 - 1):
        I_x_squared[x, y] = np.sum(Ix[x-window_size//2:x+window_size//2+1, y-window_size//2:y+window_size//2+1]**2)
        I_y_squared[x, y] = np.sum(Iy[x - window_size // 2:x + window_size // 2 + 1, y - window_size // 2:y + window_size // 2 + 1] ** 2)
        I_x_y[y, x] = np.sum(Ix[x-window_size//2:x+window_size//2+1, y-window_size//2:y+window_size//2+1]*Iy[x - window_size // 2:x + window_size // 2 + 1, y - window_size // 2:y + window_size // 2 + 1])












