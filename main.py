import cv2
import numpy as np
img = cv2.imread('img.jpg')

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red1 = np.array([0,0,0])
upper_red1 = np.array([15,255,255])

lower_red2 = np.array([150,0,0])
upper_red2 = np.array([180,255,255])

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

mask = cv2.bitwise_or(mask1, mask2)

mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))

cv2.imshow('1', mask)

mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))

cv2.imshow('2', mask)

cv2.waitKey(0)
cv2.destroyAllWindows()


res = cv2.bitwise_and(img, img, mask = mask)

cv2.imshow('a', res)

cv2.waitKey(0)
cv2.destroyAllWindows()
