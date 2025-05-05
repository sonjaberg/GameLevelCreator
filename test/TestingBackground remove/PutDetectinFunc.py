import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2 as cv
import numpy as np

# Turn on Laptop's webcam
cap = cv.VideoCapture(0)

#All colours except black/white and super desaturated
low_range = np.array([0,150,20])
high_range = np.array([179,255,255])

#Only detect white/grey
# low_range = np.array([0,0,100])
# high_range = np.array([179,50,255])

def detectObjects(targetFrame,lowRange,highRange):
    hsv = cv.cvtColor(targetFrame,cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv,lowRange,highRange)

    #Contours
    frameContor = frame.copy()
    contours, hierarchy = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    area = sorted(contours, key=cv.contourArea, reverse=True)[:5]
    cv.drawContours(frameContor,area,-1,(0,255,255),3)
    return frameContor 

#Find the largest contour::: might not need this for object detection
    # objects = []  # Store the contour we find here
    # for c in area:
    #     ep = 0.02 * cv.arcLength(c, True)
    #     corners = cv.approxPolyDP(c,ep, True)
    #     if len(corners) == 4:
    #         objects.append(c)  # Store the contour
    
#Corner detection

    # dispCorners = targetFrame.copy()
    # dispCorners = cv.drawContours(dispCorners, target_contour, -1,(0, 255, 255), 3)
    # cv.drawContours(dispCorners, corners, -1, (0, 255, 0), 10)

    #     # Sorting the corners and converting them to desired shape.
    # corners = sorted(np.concatenate(corners).tolist())

    # coords = np.zeros((4,2), dtype=np.int32)
    
    # # Displaying the corners and put coords in array
    # for index, cornerMark in enumerate(corners):
    #     coords[index, :] = cornerMark
    #     character = chr(65 + index)
    #     cv.putText(dispCorners, character, tuple(cornerMark), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv.LINE_AA)
    # return dispCorners

while True:
    ret, frame = cap.read()

    
#Colour Quantization
    frameColorQuant = frame.copy()
    reshape = frameColorQuant.reshape((-1,3))
    reshape = np.float32(reshape)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_MAX_ITER, 15, 5)
    K = 10
    ret,label,center = cv.kmeans(reshape,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((frameColorQuant.shape))

#Mask coloured items
    


    output = detectObjects(res2,low_range,high_range)

         
#Show
    cv.imshow("quant", frame)
    cv.imshow('org', output)
    # cv.imshow('mask', mask)



# Wait for the 'Esc' key to break the loop
    if cv.waitKey(24) == 27:
        break


# Release the webcam and close all OpenCV windows
cap.release()
cv.destroyAllWindows()