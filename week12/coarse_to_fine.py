import cv2

img1 = cv2.imread("./bt0.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("./bt1.png", cv2.IMREAD_GRAYSCALE)

scale = 8
h, w = img1.shape[:2]
img1_half = cv2.resize(img1, dsize=(w//scale, h//scale), interpolation=cv2.INTER_CUBIC)
img2_half = cv2.resize(img2, dsize=(w//scale, h//scale), interpolation=cv2.INTER_CUBIC)

