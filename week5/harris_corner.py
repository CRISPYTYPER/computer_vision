import numpy as np
from PIL import Image
from numpy import linalg

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

# Define the window size for local computation
window_size = 5

# Initialize arrays to store components
I_x_squared = np.zeros_like(img_array)
I_y_squared = np.zeros_like(img_array)
I_x_y = np.zeros_like(img_array)

# Compute components within the window
for x in range(1 + window_size // 2, width - window_size//2 - 1):
    for y in range(1 + window_size // 2, height - window_size//2 - 1):
        I_x_squared[x, y] = np.sum(Ix[x-window_size//2:x+window_size//2+1, y-window_size//2:y+window_size//2+1]**2)
        I_y_squared[x, y] = np.sum(Iy[x - window_size // 2:x + window_size // 2 + 1, y - window_size // 2:y + window_size // 2 + 1] ** 2)
        I_x_y[y, x] = np.sum(Ix[x-window_size//2:x+window_size//2+1, y-window_size//2:y+window_size//2+1]*Iy[x - window_size // 2:x + window_size // 2 + 1, y - window_size // 2:y + window_size // 2 + 1])

# Assemble the Harris matrx
H = np.array([[I_x_squared, I_x_y],
              [I_x_y, I_y_squared]])


large_eigen = np.zeros_like(img_array)
small_eigen = np.zeros_like(img_array)

# Compute components within the window
for x in range(1 + window_size // 2, width - window_size//2 - 1):
    for y in range(1 + window_size // 2, height - window_size//2 - 1):
        # Decompose the Harris matrix H at (x,y) into eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(H[:,:,x,y])

        # Sort eigenvalues in descending order
        eigenvalues = np.sort(eigenvalues)[::-1]
        large_eigen[x][y] = eigenvalues[0]
        small_eigen[x][y] = eigenvalues[1]

large_eigen_img = Image.fromarray(large_eigen.astype(np.uint8))
small_eigen_img = Image.fromarray(small_eigen.astype(np.uint8))
large_eigen_img.show()
small_eigen_img.show()

