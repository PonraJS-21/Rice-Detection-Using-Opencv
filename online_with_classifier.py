import cv2 as cv
import numpy as np
# import the necessary packages
from imutils.contours import sort_contours
from rice_classifier import classify_image
import numpy as np
import argparse
import imutils
import cv2
from tqdm import tqdm
import time

img_src = "images/test.png"
# load input images for demonstration
input_rice = cv.imread(img_src, cv.IMREAD_GRAYSCALE)

# local adaptive thresholding - computes local threshold based on given window size
output_adapthresh = cv.adaptiveThreshold (input_rice, 255.0,
		cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 51, -20.0)
cv.imshow("Adaptive Thresholding", output_adapthresh)
#cv.imwrite('rice_adapthresh.png', output_adapthresh)

# morphologial erosion - cleaning up binary images
kernel = np.ones((5,5),np.uint8)
output_erosion = cv.erode(output_adapthresh, kernel)
cv.imshow("Morphological Erosion", output_erosion)
#cv.imwrite('rice_erosion.png', output_erosion)

# connected components - counts and marks number of distinct foreground objects
# apply connected components on clean binary image
label_image = output_erosion.copy()
label_count = 0
rows, cols = label_image.shape
for j in range(rows):
    for i in range(cols):
        pixel = label_image[j, i]
        if 255 == pixel:
            label_count += 1
            cv.floodFill(label_image, None, (i, j), label_count)

print("Number of foreground objects", label_count)
cv.imshow("Connected Components", label_image)
#cv.imwrite('rice_components.png', label_image)

# Contours - Computes polygonal contour boundary of foreground objects
# apply connected components on clean binary image
contours, _ = cv.findContours(output_erosion, cv.RETR_EXTERNAL,  cv.CHAIN_APPROX_SIMPLE)
output_contour = cv.cvtColor(input_rice, cv.COLOR_GRAY2BGR)
cv.drawContours(output_contour, contours, -1, (0, 0, 255), 2)
print("Number of detected contours", len(contours))
cv.imshow("Contours", output_contour)
#cv.imwrite('rice_contours.png', output_contour)

# wait for key press
cv.waitKey(0)
cv.destroyAllWindows()


chars = []

font = cv2.FONT_HERSHEY_SIMPLEX 
# org 
org = (50, 50) 
# fontScale 
fontScale = 0.4 
# Blue color in BGR 
color = (0, 255, 0) 
# Line thickness of 2 px 
thickness = 1


copy_img = cv.imread(img_src)
copy_img1 = copy_img.copy()

for c in tqdm(contours):
	# compute the bounding box of the contour
	(x, y, w, h) = cv2.boundingRect(c)
	crop_img = copy_img[y:y+h, x:x+w]
	crop_img = cv2.resize(crop_img, (50, 50))
	print(crop_img.shape)
	label, probability= classify_image(crop_img)
	probability = round(probability, 2)
	if label == 'rice':
		cv2.rectangle(copy_img1, (x, y), (x + w, y + h), (0, 255, 0), 2)
		image = cv2.putText(crop_img, str(probability) + ' ' +label, org, font, fontScale, color, thickness, cv2.LINE_AA) 
		cv2.imwrite('real_data/'+str(time.time()) + str(probability) + '.jpg', image)
		cv2.imshow("cropped", image)
		# cv2.waitKey(10)

cv2.imshow("Image", copy_img1)
cv2.waitKey(0)
cv2.destroyAllWindows()