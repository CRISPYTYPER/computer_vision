import numpy as np
from PIL import Image

sigma_r = 3
patch_size = 7
patch_half = patch_size // 2
window_size = 21
window_half = window_size // 2

img = Image.open("lena_noisy.png")
img.show()

img_arr = np.array(img)
(height, width, channel) = img_arr.shape
print(height, width, channel)
denoised_img = np.zeros((height, width, channel))

# for checking runtime purpose
total_time_proportion = channel * (width - window_size) * (height - window_size)
time_sum_proportion = 0

for c in range(channel):
    for x in range(window_half, width - window_half):
        for y in range(window_half, height - window_half):
            nlm_window = img_arr[y - window_half:y + window_half + 1, x - window_half:x + window_half + 1, c]
            center_idx = (y,x,c)
            center_patch = img_arr[y-patch_half:y+1+patch_half, x-patch_half:x+1+patch_half, c]  # in this case, 7x7
            denominator_sum = 0
            numerator_sum = 0
            for i in range(window_size + 1 - patch_size):  # y (in this case, 0~14)
                for j in range(window_size + 1 - patch_size):  # x (in this case, 0~14)
                    temp_center_idx = (y-window_half+patch_half+i, x-window_half+patch_half+j,c)
                    temp_patch = img_arr[temp_center_idx[0]-patch_half:temp_center_idx[0]+1+patch_half, temp_center_idx[1]-patch_half:temp_center_idx[1]+1+patch_half,c]
                    w_p_qi = float(np.sum((center_patch - temp_patch) ** 2))
                    g_w_p_qi = np.exp(-(w_p_qi**2)/(sigma_r**2))  # 이 부분 문제의 소지가 있음 나중에 둘다 해볼것
                    # g_w_p_qi = np.exp(-(w_p_qi)/(sigma_r**2)) -> PSNR: 38.940436223129296 dB
                    # g_w_p_qi = np.exp(-(w_p_qi ** 2) / (sigma_r ** 2)) -> PSNR: 38.940436223129296 dB
                    denominator_sum += g_w_p_qi
                    numerator_sum += g_w_p_qi * img_arr[temp_center_idx]
            denoised_img[y][x][c] = numerator_sum / denominator_sum
            if denoised_img[y][x][c] > 255:
                denoised_img[y][x][c] = 255
            elif denoised_img[y][x][c] < 0:
                denoised_img[y][x][c] = 0
            time_sum_proportion += 1
            print(f"{time_sum_proportion / total_time_proportion * 100: .3f}% done")



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