import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
print('Starting Key Test')
camera=cv2.VideoCapture(0)
while(1):
    ret,frame=camera.read()
    cv2.imshow('Camera',frame)
    k = cv2.waitKey(33)
    if k==27:    # Esc key to stop 
        break
    elif k==-1:  # normally -1 returned,so don't print it
        continue
    else:
        print (k) # else print its value