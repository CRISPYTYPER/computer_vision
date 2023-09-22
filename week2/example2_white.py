import numpy as np
from PIL import Image

img = Image.open("7577.png")
img.show()

print(img.size)

(width, height) = img.size
img = np.array(img) #see the difference b.t.w np.asarray(.) and np.array(.)

for x in range(width):
    for y in range(height):
        # print((np.asarray(img)[j][i]), '\t', end='')
        img[x][y] = 255
    print()

img = Image.fromarray(img)
img.show()