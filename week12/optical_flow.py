import cv2
import numpy as np
from scipy import signal

img0 = cv2.imread("sphere0.png")
img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
img1 = cv2.imread("sphere1.png")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

img0 = img0 / 255 # normalize pixels
img1 = img1 / 255

kernel_x = np.array([[-1., 1.], [-1., 1.]])
kernel_y = np.array([[-1., -1.], [1., 1.]])
kernel_t = np.array([[1., 1.], [1., 1.]]) # *.25

fx = signal.convolve2d(img0, kernel_x, boundary='symm', mode='same')
fy = signal.convolve2d(img0, kernel_y, boundary='symm', mode='same')
ft = img0 - img1 # 예제와 같이 색깔이 나오려면 순서 바꿔야함