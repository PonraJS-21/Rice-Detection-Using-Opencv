# import the necessary packages
from imutils.contours import sort_contours
from rice_classifier import classify_image
import numpy as np
import argparse
import imutils
import cv2
from tqdm import tqdm
import time

img = cv2.imread('images/black.png')
img = cv2.resize(img, (500, 500))
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blurred =  cv2.GaussianBlur(imgray, (5, 5), 0)
# blurred = cv2.medianBlur(imgray, 5)
blurred = cv2.bilateralFilter(imgray,6 ,60 , 60)
edged = cv2.Canny(blurred, 30, 120)
ret, thresh = cv2.threshold(edged, 30, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

# cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 3)

# cv2.imshow("edge", edged)
cv2.imshow("Ground image", img)
# cv2.imshow("thresh", thresh)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method="left-to-right")[0]

# initialize the list of contour bounding boxes and associated
# characters that we'll be OCR'ing
chars = []

font = cv2.FONT_HERSHEY_SIMPLEX 
# org 
org = (50, 50) 
# fontScale 
fontScale = 0.4 
# Blue color in BGR 
color = { 'broken' : (0, 0, 255), 'un_broken' : (0, 255, 0), 'black' : (255, 0, 0) }
# Line thickness of 2 px 
thickness = 1


copy_img = img.copy()

for c in tqdm(cnts):
	# compute the bounding box of the contour
	(x, y, w, h) = cv2.boundingRect(c)
	x -= 6
	y -= 6
	w += 12
	h += 12
	if x < 0:
		x += 6
	if y < 0:
		y += 6
	crop_img = copy_img[y:y+h, x:x+w]
	label, probability= classify_image(crop_img)
	probability = round(probability, 2)
	if label == 'broken' or label == 'un_broken' or label == 'black':
		cv2.rectangle(img, (x, y), (x + w, y + h), color[label], 2)
		image = cv2.putText(crop_img, str(probability) + ' ' +label, org, font, fontScale, color[label], thickness, cv2.LINE_AA) 
		image = cv2.resize(image, (50, 50))
		cv2.imwrite('real_data/'+str(time.time()) + str(probability) + label + '.jpg', image)
		# cv2.imshow("cropped", image)
		# cv2.waitKey(10)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()