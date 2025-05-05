import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2 as cv
import numpy as np

# Turn on Laptop's webcam
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    #Code goes here
    mask = np.zeros(frame.shape[:2], np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (20,20,frame.shape[1]-20,frame.shape[0]-20)
    cv.grabCut(frame,mask,rect,bgdModel,fgdModel,4,cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    frame = frame*mask2[:,:,np.newaxis]

    #Show
    cv.imshow('frame', frame)

    # Wait for the 'Esc' key to break the loop
    if cv.waitKey(24) == 27:
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv.destroyAllWindows()
