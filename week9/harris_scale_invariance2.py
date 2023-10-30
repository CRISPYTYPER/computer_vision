import numpy as np
from PIL import Image
from numpy import linalg
import scipy
import matplotlib.pyplot as plt

img = Image.open("scale2.png").convert("L")
img.show()

img_array = np.array(img)
print(img_array.shape)

(height, width) = img_array.shape
# edge_img = np.zeros((width, height))

filter_x = np.zeros((3, 3))
filter_x[1, 1:3] = [1, -1]


filter_y = np.zeros((3, 3))
filter_y[1:3, 1] = [1, -1]

#arrays to store gradients Ix and Iy

Ix = scipy.signal.convolve2d(img_array, filter_x, boundary='fill', fillvalue=0)
Iy = scipy.signal.convolve2d(img_array, filter_y, boundary='fill', fillvalue=0)

# Define the window size for local computation
window_size = 5


target_pxl = (121, 185) # for img2
x = target_pxl[1]
y = target_pxl[0]

harris_response_arr = []
window_size_arr = []

for window_size in range(3, 30, 2):
    # Compute components within the window

    I_x_squared = np.mean(Ix[y-window_size//2:y+window_size//2+1, x-window_size//2:x+window_size//2+1]**2)
    I_y_squared = np.mean(Iy[y - window_size // 2:y + window_size // 2 + 1, x - window_size // 2:x + window_size // 2 + 1] ** 2)
    I_x_y = np.mean(Ix[y-window_size//2:y+window_size//2+1, x-window_size//2:x+window_size//2+1]*Iy[y - window_size // 2:y + window_size // 2 + 1, x - window_size // 2:x + window_size // 2 + 1])

    # Assemble the Harris matrx
    H = np.array([[I_x_squared, I_x_y],
                  [I_x_y, I_y_squared]])


    eigenvalues, eigenvectors = np.linalg.eig(H)

    # Sort eigenvalues in descending order
    eigenvalues = np.sort(eigenvalues)[::-1]
    large_eigen = eigenvalues[0]
    small_eigen = eigenvalues[1]

    img_trace = large_eigen + small_eigen
    img_det = large_eigen * small_eigen
    harris_response = img_det / img_trace
    harris_response_arr.append(harris_response)
    window_size_arr.append(window_size)
    print(harris_response)

plt.stem(window_size_arr, harris_response_arr)
plt.show()