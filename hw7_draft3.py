import cv2
import numpy as np

# Load the images
image1 = cv2.imread('IMG_8833.jpg')
image2 = cv2.imread('IMG_8834.jpg')

image1 = cv2.resize(image1, (int(image1.shape[1]/4), int(image1.shape[0]/4)))
image2 = cv2.resize(image2, (int(image2.shape[1]/4), int(image2.shape[0]/4)))

# Initialize the feature detector and extractor (e.g., SIFT)
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for both images
keypoints1, descriptors1 = sift.detectAndCompute(image2, None)
keypoints2, descriptors2 = sift.detectAndCompute(image1, None)

# Initialize the feature matcher using brute-force matching
bf = cv2.BFMatcher()

# Match the descriptors using brute-force matching
matches = bf.match(descriptors1, descriptors2)

# Select the top N matches
num_matches = 50
matches = sorted(matches, key=lambda x: x.distance)[:num_matches]

# Extract matching keypoints
src_points = np.float32([keypoints1[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
dst_points = np.float32([keypoints2[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

# Estimate the homography matrix
homography, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

# Warp the first image using the homography
result = cv2.warpPerspective(image2, homography, (image1.shape[1], image1.shape[0]))
cv2.imwrite("result.jpg", result)

# Blending the warped image with the second image using alpha blending
alpha = 0.5  # blending factor
blended_image = cv2.addWeighted(result, alpha, image1, 1 - alpha, 0)

# Display the blended image
cv2.imwrite('blended1_2_test.jpg', blended_image)

# Load the images
image3 = cv2.imread('IMG_8835.jpg')

image3 = cv2.resize(image1, (int(image1.shape[1]/4), int(image1.shape[0]/4)))

# Initialize the feature detector and extractor (e.g., SIFT)
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for both images
keypoints1, descriptors1 = sift.detectAndCompute(image3, None)
keypoints2, descriptors2 = sift.detectAndCompute(blended_image, None)

# Initialize the feature matcher using brute-force matching
bf = cv2.BFMatcher()

# Match the descriptors using brute-force matching
matches = bf.match(descriptors1, descriptors2)

# Select the top N matches
num_matches = 50
matches = sorted(matches, key=lambda x: x.distance)[:num_matches]

# Extract matching keypoints
src_points = np.float32([keypoints1[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
dst_points = np.float32([keypoints2[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

# Estimate the homography matrix
homography, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

# Warp the first image using the homography
result = cv2.warpPerspective(image3, homography, (image2.shape[1], image2.shape[0]))
cv2.imwrite("result.jpg", result)

# Blending the warped image with the second image using alpha blending
alpha = 0.5  # blending factor
blended_image2 = cv2.addWeighted(result, alpha, blended_image, 1 - alpha, 0)

# Display the blended image
cv2.imwrite('final_panorama_azfal.jpg', blended_image2)

