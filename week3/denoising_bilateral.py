from PIL import Image, ImageFilter
import numpy as np

def gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size))
    center = size // 2

    for x in range(size):
        for y in range(size):
            x_dist = x - center
            y_dist = y - center
            kernel[x, y] = (1/(2*np.pi*(sigma**2))*np.exp(-(x_dist**2 + y_dist**2)/(2*(sigma**2))))
    return kernel / np.sum(kernel)

def bilateral_filter(input_image, spatial_sigma, intensity_sigma):
    # Convert the image to a Numpy array
    image_array = np.array(input_image)

    # Create a 7x7 Gaussian spatial kernel
    spatial_kernel = gaussian_kernel(7, spatial_sigma)

    # Initialize the output image array
    output_array = np.zeros_like(image_array, dtype=np.uint8)

    # Iterate over each pixel in the image
    for y in range(image_array.shape[0]):
        for x in range(image_array.shape[1]):
            # Define the spatial neighborhood
            y_min = max(y - 3, 0)  # include
            y_max = min(y + 4, image_array.shape[0])  # exclude
            x_min = max(x - 3, 0)  # include
            y_max = min(x + 4, image_array.shape[1])  # include

            # Extract the spatial and intensity neighborhoods
            spatial_neighborhood = spatial_kernel[y_min - y + 3]
