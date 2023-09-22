import numpy as np
from PIL import Image

img = Image.open("lena_color.png")
img.show()
print(img.size)
img = np.array(img)
print(img.shape)

(height, width, channel) = img.shape

for x in range(width):
    for y in range(height):
        for c in range(channel):
            if img[y][x][c] * 2 <= 255:
                img[y][x][c] *= 2
            else:
                img[y][x][c] = 255


img = Image.fromarray(img)
img.show()

