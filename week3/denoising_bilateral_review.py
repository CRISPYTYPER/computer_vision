import numpy as np
from PIL import Image

kernel_size = (7, 7)
sigma_s = 5  # for spatial weighting function
sigma_r = 5  # for intensity-range weighting function

# You can change the variables above

def gaussian_kernel(sigma, size):
    center_idx = size[0] // 2  # 3 if
    kernel = np.fromfunction(
        lambda x, y: np.exp(-(((x-center_idx)**2) + (y-center_idx)**2) / (2 * sigma))
        , size
    )
    # not normalized yet
    return kernel

# 7x7 kernel
img = Image.open("lena_noisy.png")
img.show()

img_arr = np.array(img)
(height, width, channel) = img_arr.shape
print(height, width, channel)

denoised_img_arr = np.zeros((height, width, channel))  # for output denoised image

spatial_kernel = gaussian_kernel(sigma_s, kernel_size) # not normalized yet

kernel_half = kernel_size[0] // 2

for c in range(channel):
    for y in range(kernel_half, height-kernel_half):  # chk the boundary
        for x in range(kernel_half, width-kernel_half):  # chk the boundary

            local_box = img_arr[y-kernel_half:y+1+kernel_half, x-kernel_half:x+1+kernel_half, c]
            center_pxl = img_arr[y][x][c]

            intensity_range_kernel = np.exp(-(local_box - center_pxl)**2 / (2 * (sigma_r ** 2)))  # not normalized

            weight_sum = np.sum(spatial_kernel * intensity_range_kernel)

            denoised_img_arr[y][x][c] = np.sum(spatial_kernel * intensity_range_kernel * local_box) / weight_sum


# End of the main algoritm

# FLOAT 2 UINT8
denoised_img_arr = denoised_img_arr.astype(np.uint8)

# PSNR dB calculation (the bigger, the better performance)
original_data = img_arr
denoised_data = denoised_img_arr

# Calculate the MSE
mse = np.mean((original_data - denoised_data) ** 2)

# Calculate the maximum pixel value
max_pixel_value = np.max(original_data)

# Calulate PSNR in dB
psnr = 20 * np.log10(max_pixel_value) - 10 * np.log10(mse)

print(f"PSNR: {psnr} dB")

denoised_img = Image.fromarray(denoised_img_arr)
denoised_img.show()







