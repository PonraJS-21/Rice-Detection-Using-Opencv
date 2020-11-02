import numpy as np
import cv2

img = cv2.imread('all/IMG_20201101_183120.jpg')
img = cv2.resize(img, (500, 500))
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blurred =  cv2.GaussianBlur(imgray, (5, 5), 0)
# blurred = cv2.medianBlur(imgray, 5)
blurred = cv2.bilateralFilter(imgray,6 ,60 , 60)
edged = cv2.Canny(blurred, 10, 200)
ret, thresh = cv2.threshold(edged, 30, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours = " + str(len(contours)))
print(contours[0])

cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
cv2.drawContours(imgray, contours, -1, (0, 255, 0), 3)

# cv2.imwrite('bilateralFilter_blurred.jpg', img)
cv2.imshow('Blur', blurred)
cv2.imshow('Image', img)
cv2.imshow('Canny', edged)
cv2.waitKey(0)
cv2.destroyAllWindows()