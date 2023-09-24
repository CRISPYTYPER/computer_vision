import numpy as np
from PIL import Image


# 3x3 1/9
img = Image.open("lena_noisy.png")

img = np.array(img)
(height, width, channel) = img.shape
print(height, width, channel)
denoised_img = np.zeros((height, width, channel))

box_filter = np.ones((11,11))/121
box_filter = np.flip(box_filter, 0)
box_filter = np.flip(box_filter, 1)

for c in range(channel):
    for x in range(5, width-5):  # chk the boundary
        for y in range(5, height-5):  # chk the boundary

            tmp = img[y-5:y+6, x-5:x+6, c]  # 3x3 box
            local_mean = np.sum(box_filter * tmp)
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