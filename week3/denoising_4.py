import numpy as np
from PIL import Image


# 7x7 1/49
img = Image.open("lena_noisy.png")

img = np.array(img)
(height, width, channel) = img.shape
print(height, width, channel)
denoised_img = np.zeros((height, width, channel))

for c in range(channel):
    for x in range(3, width-3):  # chk the boundary
        for y in range(3, height-3):  # chk the boundary

            tmp = img[y-3:y+4, x-3:x+4, c]  # 7x7 box
            local_mean = np.sum(tmp)/49
            denoised_img[y][x][c] = local_mean


# FLOAT 2 UINT8
denoised_img = denoised_img.astype(np.uint8)

# PSNR dB calculation
# Convert the images to NumPy arrays
original_data = img
denoised_data = denoised_img

# Calculate the Mean Squared Error (MSE)
mse = np.mean((original_data - denoised_data) ** 2)

# Calculate the maximum pixel value
max_pixel_value = np.max(original_data)

# Calculate PSNR in dB
psnr = 20 * np.log10(max_pixel_value) - 10 * np.log10(mse)

print(f"PSNR: {psnr} dB")


denoised_img = Image.fromarray(denoised_img)
denoised_img.show()