import numpy as np
from PIL import Image

img = Image.open("lena_color.png")
print(img.size)

img = img.resize((640, 480))
img.show()

img = np.array(img)
print(img.shape)

(height, width, channel) = img.shape
denoised_img = np.zeros((height, width, channel))

ix_filter = np.zeros((3,3))
ix_filter[1,1] = 2
ix_filter = ix_filter - (np.ones((3,3)) / 9)

for c in range(channel):
    for x in range(1, width-1):
        for y in range(1, height-1):
            tmp = img[y - 1:y + 2, x - 1:x + 2, c] # 3x3box
            denoised_img[y][x][c] = np.sum(ix_filter*tmp)
            if denoised_img[y][x][c] > 255:
                denoised_img[y][x][c] = 255
            elif denoised_img[y][x][c] < 0:
                denoised_img[y][x][c] = 0



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
