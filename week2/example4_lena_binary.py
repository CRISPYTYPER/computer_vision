import numpy as np
from PIL import Image

img = Image.open("lena_color.png")
img.show()

print(img.size)
img = np.array(img)
print(img.shape)

(height, width, channel) = img.shape
gray_img = np.zeros(shape=[height, width], dtype=np.uint8)

for x in range(width):
    for y in range(height):
        sum_rgb = 0
        for c in range(channel):
            sum_rgb += img[x][y][c]
        gray_img[x][y] = sum_rgb / 3
    print()

img = Image.fromarray(gray_img)
img.show()

img = np.array(img)
(width, height) = img.shape

for x in range(width):
    for y in range(height):
        if img[x][y] > 127:
            img[x][y] = 255
        else:
            img[x][y] = 0
    print()

img = Image.fromarray(img)
img.show()