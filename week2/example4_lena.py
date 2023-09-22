import numpy as np
from PIL import Image

img = Image.open("lena_color.png")
img.show()
print(img.size)
img = np.array(img)
print(img.shape)