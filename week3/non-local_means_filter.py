import numpy as np
from PIL import Image, ImageFilter


# 5x5 gaussian
img = Image.open("lena_noisy.png")

# Create a 5x5 Gaussian kernel
kernel_size = (5, 5)
sigma = 1.0
def gaussian_kernel(sigma):
    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma ** 2)) *
                     np.exp(-((x - (kernel_size[0] - 1) / 2) ** 2 + (y - (kernel_size[1] - 1) / 2) ** 2) / (2 * sigma ** 2)),
        kernel_size
    )
    kernel /= np.sum(kernel)  # Normalize the kernel; make the sum to 1
    return kernel

img = np.array(img)
(height, width, channel) = img.shape
print(height, width, channel)
for i in range(1, 6):
    current_kernel = gaussian_kernel(i)
    denoised_img = np.zeros((height, width, channel), dtype=np.uint8)

    for c in range(channel):
        for x in range(2, width-2):  # chk the boundary
            for y in range(2, height-2):  # chk the boundary

                tmp = img[y-2:y+3, x-2:x+3, c]  # 5x5 box
                tmp = tmp * current_kernel
                denoised_img[y][x][c] = np.sum(tmp)


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