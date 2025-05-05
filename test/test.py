import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2 as cv
import numpy as np

# Turn on Laptop's webcam
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
 
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    
    corners = cv.goodFeaturesToTrack(gray,10,0.001,50)
    corners = np.int32(corners)
    
    for i in corners:
        x,y = i.ravel()
        cv.circle(frame,(x,y),3,255,-1)
 
#Show
    cv.imshow('frame', frame)


    # Wait for the 'Esc' key to break the loop
    if cv.waitKey(24) == 27:
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv.destroyAllWindows()
