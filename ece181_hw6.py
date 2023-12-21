import numpy as np
import cv2
import matplotlib.pyplot as plt

def harris_corner_detection(image, max_corners=50, threshold=0.01):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Apply Gaussian blur to the grayscale image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Calculate derivatives using Sobel operators after Gaussian smoothing
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # Compute elements of the structure matrix
    Ix2 = sobel_x ** 2
    Iy2 = sobel_y ** 2
    Ixy = sobel_x * sobel_y

    # Apply another Gaussian blur to the structure matrix elements
    k_size = 5
    Ix2_blurred = cv2.GaussianBlur(Ix2, (k_size, k_size), 0)
    Iy2_blurred = cv2.GaussianBlur(Iy2, (k_size, k_size), 0)
    Ixy_blurred = cv2.GaussianBlur(Ixy, (k_size, k_size), 0)

    height, width = gray.shape
    corner_response = np.zeros((height, width))

    # Compute Harris corner response for each pixel
    det_M = Ix2_blurred * Iy2_blurred - Ixy_blurred ** 2
    trace_M = Ix2_blurred + Iy2_blurred
    # Choose k = 0.04
    corner_response = det_M - 0.04 * (trace_M ** 2)

    # Apply threshold to get strong corners
    corner_response[corner_response < threshold * corner_response.max()] = 0

    # Find coordinates of top N corners
    corner_coords = np.argpartition(corner_response.flatten(), -max_corners)[-max_corners:]
    top_corners = [(coord // width, coord % width) for coord in corner_coords]

    # Draw circles around detected corners on the image
    image_with_corners = np.copy(image)
    for corner in top_corners:
        cv2.circle(image_with_corners, (corner[1], corner[0]), 5, (0, 255, 0), 2)  # (x, y) reversal for cv2.circle

    return top_corners, image_with_corners

# Function to create a Gaussian kernel
def gaussian_kernel(size, sigma):
    # construct kernel then take outer product to get 2D kernel
    kernel = cv2.getGaussianKernel(size, sigma)
    kernel_2d = np.outer(kernel, kernel.transpose())
    return kernel_2d

# Function for applying Gaussian smoothing to an image
def gaussian_filter(image, kernel):
    smoothed = cv2.filter2D(image, -1, kernel)
    return smoothed

# Build the Gaussian pyramid manually
def gaussian_pyramid(image, size=5, sigma=1.4, levels=3):
    image = image[:512, :512] # trim to appropriate dimensions
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pyramid = [gray_image]
    for _ in range(levels):
        kernel = gaussian_kernel(size, sigma)  # Adjust kernel size and sigma as needed
        smoothed = gaussian_filter(pyramid[-1], kernel)
        downsampled = smoothed[::2, ::2]  # Subsample by a factor of 2
        pyramid.append(downsampled)
    return pyramid

# Build the Laplacian pyramid manually
def laplacian_pyramid(image, size=5, sigma=1.4, levels=4):
    image = image[:512, :512]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pyramid = [gray_image]
    for _ in range(levels):
        kernel = gaussian_kernel(size, sigma)  # Adjust kernel size and sigma as needed
        smoothed = gaussian_filter(pyramid[-1], kernel)
        # take diff of gaussian smoothed with previously downsized image in pyramid
        diff_of_gaussians = smoothed - pyramid[-1]
        pyramid.append(diff_of_gaussians)
        downsampled = diff_of_gaussians[::2, ::2]  # Subsample by a factor of 2
        pyramid.append(downsampled)
    return pyramid

# Read an image
image_path1 = "left.jpg"
image_path2 = "right.jpg"
input_image = cv2.imread(image_path2)

# Apply Harris corner detection
#detected_corners, image_with_corners = harris_corner_detection(input_image, max_corners=50)
#print(detected_corners)
# Display the pyramid images
laplacian_pyr = laplacian_pyramid(input_image)

# Display each level of the pyramid with adjusted plot sizes
plt.figure(figsize=(laplacian_pyr[0].shape[1] / 100, laplacian_pyr[0].shape[0] / 100))
plt.imshow(laplacian_pyr[0], cmap='gray')
plt.title(f"Level 0 - {laplacian_pyr[0].shape}")
plt.axis('off')
plt.show()

plt.figure(figsize=(laplacian_pyr[1].shape[1] / 100, laplacian_pyr[1].shape[0] / 100))
plt.imshow(laplacian_pyr[1], cmap='gray')
plt.title(f"Level 1 - {laplacian_pyr[1].shape}")
plt.axis('off')
plt.show()

plt.figure(figsize=(laplacian_pyr[2].shape[1] / 100, laplacian_pyr[2].shape[0] / 100))
plt.imshow(laplacian_pyr[2], cmap='gray')
plt.title(f"Level 2 - {laplacian_pyr[2].shape}")
plt.axis('off')
plt.show()

plt.figure(figsize=(laplacian_pyr[3].shape[1] / 100, laplacian_pyr[3].shape[0] / 100))
plt.imshow(laplacian_pyr[3], cmap='gray')
plt.title(f"Level 3 - {laplacian_pyr[3].shape}")
plt.axis('off')
plt.show()

plt.figure(figsize=(laplacian_pyr[4].shape[1] / 100, laplacian_pyr[4].shape[0] / 100))
plt.imshow(laplacian_pyr[4], cmap='gray')
plt.title(f"Level 3 - {laplacian_pyr[4].shape}")
plt.axis('off')
plt.show()

'''
plt.figure(figsize=(8, 6))
for i, level in enumerate(pyramid):
    plt.subplot(1, len(pyramid), i + 1)
    plt.imshow(level, cmap='gray')
    plt.title(f'Level {i}')
    plt.axis('off')
plt.tight_layout()
plt.show()
'''
# Display the detected corners
'''
cv2.imshow('Harris Corners', image_with_corners)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
