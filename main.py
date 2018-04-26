import cv2
import numpy as np
img = cv2.imread('img.jpg')

lower_red1 = np.array([0,0,0])
upper_red1 = np.array([0,0,0])

lower_red2 = np.array([170, 170, 100])
upper_red2 = np.array([180, 255, 255])

minred = np.array([255,255,255])
maxred = np.array([0,0,0])


global hsv

cap = cv2.VideoCapture(0)

def onclick(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN : 
        global hsv
        pixel = hsv[y,x]
        print(pixel)
        for i, c in enumerate(pixel) :
            if c > maxred[i] : maxred[i] = c
            if c < minred[i] : minred[i] = c

cv2.namedWindow('frame')
cv2.setMouseCallback('frame', onclick)



while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, minred, maxred)

    mask = cv2.bitwise_or(mask1, mask2)

    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))

    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))

    res = cv2.bitwise_and(frame, frame, mask = mask)

    x, y = (mask > 0).nonzero()
    if len(x) > 0 :
        mx = int(sum(x)/len(x))
        my = int(sum(y)/len(y))
        cv2.circle(frame, (my, mx), 4, 255, 3)


    cv2.imshow('frame', frame)
    cv2.imshow('res', res)
    cv2.imshow('mask', mask)

    k = cv2.waitKey(5) & 0xFF
    if k == 27 : break

cv2.destroyAllWindows()
cap.release()