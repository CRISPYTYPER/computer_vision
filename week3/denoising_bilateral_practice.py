import numpy as np
from PIL import Image

kernel_size = (7, 7)
sigma = 5.0
def gaussian_kernel(sigma):
    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma ** 2)) *
                     np.exp(-((x - (kernel_size[0] - 1) / 2) ** 2 + (y - (kernel_size[1] - 1) / 2) ** 2) / (2 * sigma ** 2)),
        kernel_size
    )
    kernel /= np.sum(kernel)  # Normalize the kernel; make the sum to 1
    return kernel

# 7x7 1/49
img = Image.open("lena_noisy.png")
img.show()

img = np.array(img)
(height, width, channel) = img.shape
print(height, width, channel)


denoised_img = np.zeros((height, width, channel))

filter_size = 7
filter_size_half = filter_size//2

# spatial_weight = np.ones((7,7)) # 7x7 gaussian
spatial_weight = gaussian_kernel(sigma)

# Create a 5x5 Gaussian kernel

sigma_r = 5
for c in range(channel):
    for x in range(filter_size_half, width-filter_size_half): #chk the boundary
        for y in range(filter_size_half, height-filter_size_half): #chk the boundary

            local_data = img[y-filter_size_half:y+filter_size_half+1, x-filter_size_half:x+filter_size_half+1, c]
            ref = img[y][x][c]

            range_weight = np.exp(-(local_data - ref)**2/sigma_r**2)

            weights = range_weight * spatial_weight
            normalization = np.sum(weights)

            denoised_img[y][x][c] = np.sum(weights*local_data)/normalization


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