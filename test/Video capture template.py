import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2 as cv

# Turn on Laptop's webcam
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    #Code goes here
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)

    #Show
    cv.imshow('frame', frame)
    cv.imshow('frame', hsv)

    # Wait for the 'Esc' key to break the loop
    if cv.waitKey(24) == 27:
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv.destroyAllWindows()
