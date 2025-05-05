import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2 as cv
import numpy as np

low_range = np.array([50-90,50,50])
high_range = np.array([135-90,255,255])

bg = np.zeros((600,600,3), np.uint8)


# Turn on Laptop's webcam
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    #Code goes here
    mask = cv.inRange(hsv,low_range,high_range)
    #Show
    cv.imshow('frame', mask)
    cv.imshow('org', frame)

    # Wait for the 'Esc' key to break the loop
    if cv.waitKey(24) == 27:
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv.destroyAllWindows()
