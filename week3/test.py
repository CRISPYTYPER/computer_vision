import numpy as np

def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma ** 2)) *
                     np.exp(-((x - (size-1) / 2) ** 2 + (y - (size-1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    kernel /= np.sum(kernel)
    return kernel

# Create a 5x5 Gaussian kernel with sigma = 1.0
size = 5
sigma = 1.0
gaussian_5x5 = gaussian_kernel(size, sigma)

print("5x5 Gaussian Kernel:")
print(gaussian_5x5)
print("Sum of Kernel Elements:", np.sum(gaussian_5x5))
