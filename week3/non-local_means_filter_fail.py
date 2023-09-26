import numpy as np
from PIL import Image

sigma_r = 3
mask_size = 7
mask_half = mask_size // 2
big_mask_size = 21
big_mask_half = big_mask_size // 2

img = Image.open("lena_noisy.png")
img.show()

img = np.array(img)
(height, width, channel) = img.shape
print(height, width, channel)
denoised_img = np.zeros((height, width, channel))

for c in range(channel):
    for x in range(big_mask_half, width-big_mask_half):
        for y in range(big_mask_half, height-big_mask_half):
            nlm_mask = img[y-big_mask_half:y+big_mask_half+1, x-big_mask_half:x+big_mask_half+1, c]
            middle_mask = nlm_mask[big_mask_half-mask_half:big_mask_half-mask_half+mask_size, big_mask_half-mask_half:big_mask_half-mask_half+mask_size]
            numerator = 0
            denominator = 0

            for k in range(mask_half, big_mask_size - mask_half): #x
                for m in range(mask_half, big_mask_size - mask_half):#y
                    temp_mask = nlm_mask[m-mask_half: m+mask_half+1, k-mask_half: k+mask_half+1]
                    w = np.sum((middle_mask-temp_mask)**2)
                    gaussianed = np.exp(-(w**2)/(sigma_r**2))
                    numerator += (gaussianed * nlm_mask[m][k])
                    denominator += gaussianed

            denoised_img[y][x][c] = numerator/denominator
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