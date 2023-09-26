import numpy as np
from PIL import Image

img = Image.open("lena_color.png")
print(img.size)

# img = img.resize((640, 480))

img = np.array(img)
print(img.shape)

(height, width, channel) = img.shape
denoised_img = np.zeros((height, width, channel))

ix_filter = np.zeros((3,3))
ix_filter[:,0] = [1,1,1]
ix_filter[:,2] = [-1,-1,-1]
ix_filter= np.flip(ix_filter,axis=None)



iy_filter = np.zeros((3,3))
iy_filter[0,:] = [1,1,1]
iy_filter[2,:] = [-1,-1,-1]
iy_filter= np.flip(iy_filter, axis=None)

for c in range(channel):
    for x in range(1, width-1):
        for y in range(1, height-1):
            tmp = img[y - 1:y + 2, x - 1:x + 2, c] # 3x3box
            local_mean_ix = np.sum(ix_filter * tmp)
            local_mean_iy = np.sum(iy_filter * tmp)
            denoised_img[y][x][c] = (local_mean_ix ** 2 + local_mean_iy ** 2) ** (1 / 2)

# Normalize the filtered image to the 0-255 range
denoised_img = (denoised_img - np.min(denoised_img)) / (np.max(denoised_img) - np.min(denoised_img)) * 255

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
