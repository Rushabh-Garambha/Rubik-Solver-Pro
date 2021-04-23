import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    img_hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    img_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)


    img_h = img_hsv[:,:,0]
    img_s = img_hsv[:,:,1]
    img_v = img_hsv[:,:,2]
    print(img_s.shape)
    cv.imshow('h+s+v',np.hstack((img_h,img_s,img_v)))
    cv.imshow('bgr+hsv',np.hstack((frame,img_hsv)))
    cv.imshow('gray',img_gray)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()


