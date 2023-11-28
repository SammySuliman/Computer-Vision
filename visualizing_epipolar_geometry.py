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
    x = source[:, 0]
    y = source[:, 1]
    xp = target[:, 0]
    yp = target[:, 1]
    A = np.zeros((2*x.shape[0], 8))
    b = np.zeros((2*x.shape[0], 1))

    idx = 0

    for _x, _y, _xp, _yp in zip(x, y, xp, yp):
        A[idx, :] = [_x, _y, 1, 0, 0, 0, -_x*_xp, -_y*_xp]
        b[idx] = _xp
        idx += 1
        A[idx, :] = [0, 0, 0, _x, _y, 1, -_x * _yp, -_y * _yp]
        b[idx] = _yp
        idx += 1

    #H = np.linalg.solve(A, b)
    H = np.linalg.lstsq(A, b, rcond=None)[0]
    H = np.append(H, 1)
    H = np.reshape(H, (3, 3))
    return H

pts1 = get_points(img1, n_pts=8)
pts1 = np.array(pts1)
pts2 = get_points(img2, n_pts=8)
pts2 = np.array(pts2)
F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_8POINT, ransacReprojThreshold = 3.0)
np.savetxt('fundamental_matrix.txt', F)
print(F)

def drawEpipolarLine(x_coord, y_coord, right_image, fund_matrix = F):
    # Put point into homogeneous vector form
    homogeneous_x = np.array([x_coord, y_coord, 1])
    # Compute l' = Fx
    l_prime = np.matmul(fund_matrix, homogeneous_x)
    # Write homogenous vector as Cartesian line and calculate slope from form y=mx+b 
    m = (-1 * x_coord) / y_coord
    # Get dimensions of the image
    image_height, image_width, _ = right_image.shape
    # Find x values constrained to image size
    x_values = np.linspace(0, image_width - 1, image_width)

    # Calculate corresponding y values using the equation of the line (point-slope form)
    y_values = m * (x_values - x_coord) + y_coord
    # Constrain y values to image size
    y_values = np.clip(y_values, 0, image_height - 1)
    plt.imshow(right_image)
    plt.plot(x_values, y_values, 'g') # Plot a green line  
    plt.show()

x = get_points(img1, n_pts=1)
print(x)
drawEpipolarLine(x[0][0], x[0][1], img2)
