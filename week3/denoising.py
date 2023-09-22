import numpy as np
from PIL import Image

img = Image.open("lena_noisy.png")

img = np.array(img)
(height, width, channel) = img.shape
print(height, width, channel)
denoised_img = np.zeros((height, width, channel))

for c in range(channel):
    for x in range(1, width-1):  # chk the boundary
        for y in range(1, height-1):  # chk the boundary

            tmp = img[y-1:y+2, x-1:x+2, c]  # 3x3 box
            local_mean = np.sum(tmp)/9
            denoised_img[y][x][c] = local_mean


# FLOAT 2 UINT8
denoised_img = denoised_img.astype(np.uint8)

denoised_img = Image.fromarray(denoised_img)
denoised_img.show()