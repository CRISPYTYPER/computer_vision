import numpy as np
from PIL import Image

img = Image.open("lena_color.png")
img.show()

img = np.array(img)
(height, width, channel) = img.shape

n = np.random.normal(0, 10, (height, width, channel))

print(img.dtype, n.dtype)

img = img + n  # add random value

# CLIP
img[img > 255] = 255
img[img < 0] = 0

# FLOAT 2 UINT8
img = img.astype(np.uint8)

img = Image.fromarray(img)
img.show()

img.save("lena_noisy.png")
