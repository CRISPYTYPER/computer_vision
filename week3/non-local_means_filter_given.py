import numpy as np
from PIL import Image

img = Image.open("lena_noisy.png")

img = np.array(img)
(height, width, channel) = img.shape
print(height, width, channel)
denoised_image = np.zeros((height, width, channel))

def compute_w(patch_ref, patch_qry):
    dist = np.mean((patch_ref - patch_qry)**2)  # why mean rather than sum?
    return dist

search_window = 15
search_window_half = search_window//2
patch_size_half = 7//2  #7x7 patch will be compared

sigma_r = 5 # user parameter

for x in range(0, width):
    for y in range(0, height):
        for c in range(channel):
            try:
                patch_ref = img[y-patch_size_half:y+1+patch_size_half, x-patch_size_half:x+patch_size_half+1, c]  # need to chk out of index
                normalization = 0
                out = 0
            except IndexError:
                continue

            for xx in range(x, x+search_window):
                for yy in range(y, y+search_window):

                    if(yy-patch_size_half>=y-search_window_half and yy+patch_size_half <=y+search_window_half and xx-patch_size_half>=x-search_window_half and xx+patch_size_half <= x+search_window_half ):
                        patch_query = img[yy-patch_size_half:yy+1+patch_size_half, xx-patch_size_half:xx+1+patch_size_half, c]
                        w_pq = compute_w(patch_ref, patch_query)
                        G_w_pq = np.exp(-(w_pq) / (sigma_r ** 2))

                        I_qry = img[xx, yy, c]
                        out += G_w_pq * I_qry
                        normalization += G_w_pq



            out = out/normalization
            denoised_image[y][x][c] = out

# FLOAT 2 UINT8
denoised_image = denoised_image.astype(np.uint8)


# PSNR dB calculation
# Convert the images to NumPy arrays
original_data = img
denoised_data = denoised_image

# Calculate the Mean Squared Error (MSE)
mse = np.mean((original_data - denoised_data) ** 2)

# Calculate the maximum pixel value
max_pixel_value = np.max(original_data)

# Calculate PSNR in dB
psnr = 20 * np.log10(max_pixel_value) - 10 * np.log10(mse)

print(f"PSNR: {psnr} dB")


denoised_image = Image.fromarray(denoised_image)
denoised_image.show()