import numpy as np
from PIL import Image

img = Image.open("7577.png")
img.show()

print(img.size)

(width, height) = img.size

for j in range(width):
    for i in range(height):
        print((np.asarray(img)[j][i]), '\t', end='')
    print()