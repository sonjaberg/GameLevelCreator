import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2 as cv
import numpy as np

# Turn on Laptop's webcam
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()

    #Code goes here
    reshape = frame.reshape((-1,3))
    reshape = np.float32(reshape)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_MAX_ITER, 15, 5)
    K = 8
    ret,label,center = cv.kmeans(reshape,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((frame.shape))

    #Show
    cv.imshow('frame', res2)
    cv.imshow('frame2', frame)

    # Wait for the 'Esc' key to break the loop
    if cv.waitKey(24) == 27:
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv.destroyAllWindows()
