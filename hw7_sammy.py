import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

img1 = cv2.imread('IMG_8833.jpg')  #queryimage 
img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) #grayscale
img = cv2.resize(img, (1000, 1000))
H = np.array([[1.5, 0.5, 0],[0, 2.5, 0],[0, 0, 1]])
#print(H)
img_prime = cv2.warpPerspective(img, H, (1000, 1000))
cv2.imshow('image', img_prime)
cv2.waitKey(0)

# Initiate SIFT detector
obj = cv2.SIFT_create()
keypoints1, descriptors1 = obj.detectAndCompute(img, None)
keypoints2, descriptors2 = obj.detectAndCompute(img_prime, None)
pts1 = np.float32([kp.pt for kp in keypoints1]).reshape(-1, 1, 2)
pts2 = np.float32([kp.pt for kp in keypoints2]).reshape(-1, 1, 2)

# finding matches from BFMatcher()
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.match(descriptors1, descriptors2)
sorted_matches = sorted(matches, key = lambda x:x.distance)
top_matches = sorted_matches[:30]
#print('matches', matches)
groundtruth_mapping = cv2.perspectiveTransform(pts1, H)
#print(groundtruth_mapping)
true_match = 0
for match in top_matches:
    query = pts1[match.queryIdx]
    train = pts2[match.trainIdx]
    groundtruth = groundtruth_mapping[match.queryIdx]
    dist = np.linalg.norm(np.array(groundtruth) - np.array(train))
    if dist <= 3:
        true_match += 1 
print(true_match)
percentage_mapped_correctly = (true_match / len(top_matches)) * 100
print('Percentage mapped correctly:', percentage_mapped_correctly)


