import numpy as np
from PIL import Image

img = Image.open("checkerboard.png").convert("L")
img.show()

img_array = np.array(img)
print(img_array.shape)

(width, height) = img_array.shape
# edge_img = np.zeros((width, height))

sobel_x = np.zeros((3,3))
sobel_x[:,0] = [1,2,1]
sobel_x[:,2] = [-1,-2,-1]
sobel_x = np.flip(sobel_x,axis=None) # flip-flop


sobel_y = np.zeros((3,3))
sobel_y[0,:] = [1,2,1]
sobel_y[2,:] = [-1,-2,-1]
sobel_y= np.flip(sobel_y, axis=None) # flip-flop

# Initialize arrays to store gradients Ix and Iy
Ix = np.zeros_like(img_array)
Iy = np.zeros_like(img_array)

for x in range(1, width-1):
    for y in range(1, height-1):
        tmp = img_array[x - 1:x + 2, y - 1:y + 2] # 3x3 box
        conv_x = np.sum(sobel_x*tmp)
        conv_y = np.sum(sobel_y*tmp)
        Ix[x,y] = conv_x
        Iy[x,y] = conv_y
#
#
#
# # FLOAT 2 UINT8
# edge_img = edge_img.astype(np.uint8)
#
# # PSNR dB calculation
# # Convert the images to NumPy arrays
# original_data = img
# denoised_data = edge_img
#
# # Calculate the Mean Squared Error (MSE)
# mse = np.mean((original_data - denoised_data) ** 2)
#
# # Calculate the maximum pixel value
# max_pixel_value = np.max(original_data)
#
# # Calculate PSNR in dB
# psnr = 20 * np.log10(max_pixel_value) - 10 * np.log10(mse)
#
# print(f"PSNR: {psnr} dB")
#
#
# edge_img = Image.fromarray(edge_img)
# edge_img.show()
#
# # result: edge detected without noise, but the edge is to blurry
