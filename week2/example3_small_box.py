import numpy as np
from PIL import Image

img = Image.open("7577.png")
img.show()

print(img.size)

(width, height) = img.size
img = np.array(img) #see the difference b.t.w np.asarray(.) and np.array(.)

min_x = width - 1
min_y = height - 1
max_x = 0
max_y = 0

for x in range(width):
    for y in range(height):
        # print((np.asarray(img)[j][i]), '\t', end='')
        if img[x][y] != 0:
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y

    for x in range(width):
        for y in range(height):
            # print((np.asarray(img)[j][i]), '\t', end='')
            if x >= min_x and x <= max_x and y >= min_y and y <= max_y:
                img[x][y] = 127
    print()

img = Image.fromarray(img)
img.show()