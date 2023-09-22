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
            img[y][x][c] = pow((img[y][x][c]/255), (1/3))*255


img = Image.fromarray(img)
img.show()

