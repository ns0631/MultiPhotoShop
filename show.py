import cv2

image = cv2.imread('EyePhoto.jpg')

region = image[32:86, 53:112, :]

cv2.imshow('Cut', region)
cv2.waitKey(0)