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
            if c == 1 or c == 2:
                img[x][y][c] = 0
    print()

img = Image.fromarray(img)
img.show()
