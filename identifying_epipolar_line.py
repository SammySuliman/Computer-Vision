import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import argparse

def get_points(img, n_pts):
  plt.imshow(img)
  #plt.show()
  pts = plt.ginput(n_pts)
  plt.close()
  return pts

# Load your image
img1 = cv.imread('left.jpg')  #queryimage # left image
img2 = cv.imread('right.jpg') #trainimage # right image

def getHomography(source, target):
    # Use given function to obtain 4 points
    points = get_points(source, 4)
    # Unpack x- and y-coords in a list
    x_coords_orig = [x for x, y in points]
    y_coords_orig = [y for x, y in points]
    points2 = get_points(target, 4)
    # unpack new x, y coords into 2 lists
    x_coords_new = [x for x, y in points2]
    y_coords_new = [y for x, y in points2]
    # Manually define 8x8 matrix A based on equations given in lecture using old and new points
    A = [[x_coords_orig[0], y_coords_orig[0], 1, 0, 0, 0, -1 * x_coords_orig[0] * x_coords_new[0], -1 * y_coords_orig[0] * x_coords_new[0]],
        [0, 0, 0, x_coords_orig[0], y_coords_orig[0], 1, -1 * x_coords_orig[0] * y_coords_new[0], -1 * y_coords_orig[0] * y_coords_new[0]],
        [x_coords_orig[1], y_coords_orig[1], 1, 0, 0, 0, -1 * x_coords_orig[1] * x_coords_new[1], -1 * y_coords_orig[1] * x_coords_new[1]],
        [0, 0, 0, x_coords_orig[1], y_coords_orig[1], 1, -1 * x_coords_orig[1] * y_coords_new[1], -1 * y_coords_orig[1] * y_coords_new[1]],
        [x_coords_orig[2], y_coords_orig[2], 1, 0, 0, 0, -1 * x_coords_orig[2] * x_coords_new[2], -1 * y_coords_orig[2] * x_coords_new[2]],
        [0, 0, 0, x_coords_orig[2], y_coords_orig[2], 1, -1 * x_coords_orig[2] * y_coords_new[2], -1 * y_coords_orig[2] * y_coords_new[2]],
        [x_coords_orig[3], y_coords_orig[3], 1, 0, 0, 0, -1 * x_coords_orig[3] * x_coords_new[3], -1 * y_coords_orig[3] * x_coords_new[3]],
        [0, 0, 0, x_coords_orig[3], y_coords_orig[3], 1, -1 * x_coords_orig[3] * y_coords_new[3], -1 * y_coords_orig[3] * y_coords_new[3]]]
    A = np.array(A)
    # Manually define 8x1 vector b based on equations given in lecture using old and new points
    b = [x_coords_new[0], y_coords_new[0], x_coords_new[1], y_coords_new[1],
        x_coords_new[2], y_coords_new[2], x_coords_new[3], y_coords_new[3]]
    b = np.array(b)
    # solve least squares equation
    h = np.linalg.solve(A, b)
    h = np.append(h, 1)
    H = np.reshape(h, (3,3))
    return H

pts1 = get_points(img1, n_pts=8)
pts1 = np.array(pts1)
pts2 = get_points(img2, n_pts=8)
pts2 = np.array(pts2)
F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_8POINT, ransacReprojThreshold = 3.0)
np.savetxt('fundamental_matrix.txt', F)
print(F)

def drawEpipolarLine(x_coord, y_coord, left_image, right_image, fund_matrix=F):
    # Put Cartesian point into homogeneous vector form
    homogeneous_x = np.array([x_coord, y_coord, 1])
    # Compute epipolar line l'=Fx
    l_prime = np.matmul(fund_matrix, homogeneous_x)
    m = (-1 * x_coord) / y_coord
    # Get dimensions of the image
    image_height, image_width, _ = right_image.shape
    # x values constrained to size of image
    x_values = np.linspace(0, image_width - 1, image_width)

    # Calculate corresponding y values using the equation of the line
    y_values = m * (x_values - x_coord) + y_coord
    # Constrained to image size
    y_values = np.clip(y_values, 0, image_height - 1)

    # Create a figure and axis for the composite image
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns for side-by-side images
    # Plot the left image on the first axis
    ax[0].imshow(left_image)
    ax[0].scatter(x_coord, y_coord, c='red', marker='o', s=80, label='Point')
    ax[0].set_title('Left Image with Chosen Point x')

    # Plot the right image on the second axis
    ax[1].imshow(right_image)
    ax[1].plot(x_values, y_values, 'g')  # Plot a green line on the right image
    ax[1].set_title('Right Image with Epipolar Line')
    plt.show()

x = get_points(img1, n_pts=1)
print(x)
drawEpipolarLine(x[0][0], x[0][1], img1, img2)

H = getHomography(img2, img1)
print(H)
im21 = cv.warpPerspective(img2, H, (2000, 1500))
cv.imshow('image', im21)
cv.waitKey(0)
