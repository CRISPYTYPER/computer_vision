import numpy as np
from PIL import Image


# 3x3 1/9
img = Image.open("lena_color.png")

img = np.array(img)
(height, width, channel) = img.shape
print(height, width, channel)
pad_image = np.zeros((height+20, width+20, channel)) # 10 10 10 10 paddings


for c in range(channel):
    for x in range(width):  # chk the boundary
        for y in range(height):  # chk the boundary

            pad_image[y+10, x+10, c] = img[y, x, c]

for c in range(channel): # corners
    pad_image[0:10+1, 0:10+1, c] = img[0, 0, c] # left top corner
    pad_image[0:10+1, width+10-1:width+20, c] = img[0, width-1, c] # right top corner
    pad_image[height+10-1:height+20, 0:10+1, c] = img[height-1, 0, c] # left bottom corner
    pad_image[height+10-1:height+20, width+10-1:width+20, c] = img[height-1, width-1, c] # left bottom corner

for c in range(channel): # top bottom side
    for x in range(width):
        for y in range(10):
                pad_image[y, 10+x, c] = img[0,x,c]
                pad_image[height+10+y, 10+x, c] = img[height-1,x,c]
for c in range(channel): # left right side
    for y in range(height):
        for x in range(10):
                pad_image[y+10, x, c] = img[y,0,c]
                pad_image[y+10, 10+width+x, c] = img[y,width-1,c]

# FLOAT 2 UINT8
pad_image = pad_image.astype(np.uint8)

pad_image = Image.fromarray(pad_image)
pad_image.show()