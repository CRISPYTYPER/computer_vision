import numpy as np
from PIL import Image

img = Image.open("lena_color.png")
img.show()

print(img.size)
img = np.array(img)

print(img.shape)

(height, width, channel) = img.shape
new_img = np.zeros(shape=[width, height, channel], dtype=np.uint8)
print(new_img.shape)

for x in range(width):
    for y in range(height):
        for c in range(channel):
            new_img[y][x][c] = img[x][y][c]

img = Image.fromarray(new_img)
img.show()
