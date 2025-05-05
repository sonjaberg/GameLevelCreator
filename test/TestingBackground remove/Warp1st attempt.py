import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2 as cv
import numpy as np

# Turn on Laptop's webcam
cap = cv.VideoCapture(0)

#All colours except black/white and super desaturated
#low_range = np.array([0,150,20])
#high_range = np.array([179,255,255])
# low_range = np.array([0,150,150])
# high_range = np.array([20,255,255])

#Only detect white/grey
low_range = np.array([0,0,70])
high_range = np.array([179,50,255])

def reorder(pts):
    sum = np.sum(pts,axis=1)

    topLeft = pts[np.argmin(sum)]
    bottomRight = pts[np.argmax(sum)]

    diff = np.diff(pts,axis=1)

    topRight = pts[np.argmin(diff)]
    bottomLeft = pts[np.argmax(diff)]

    return np.array([topLeft,topRight,bottomRight,bottomLeft])

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
    hsv = cv.cvtColor(res2,cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv,low_range,high_range)

#Contours
    frameContor = frame.copy()
    contours, hierarchy = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    area = sorted(contours, key=cv.contourArea, reverse=True)[:5]
    cv.drawContours(frameContor,area,-1,(0,255,255),3)
    
#Corner detection
    #Find the largest contour
    target_contour = None  # Store the contour we find here
    for c in area:
        ep = 0.02 * cv.arcLength(c, True)
        corners = cv.approxPolyDP(c,ep, True)
        if len(corners) == 4:
            target_contour = c  # Store the contour
            break

    dispCorners = frame.copy()
    warpedImage = frame.copy()
    if target_contour is not None:  # Check if found a contour with 4 corners
        dispCorners = cv.drawContours(dispCorners, target_contour, -1,(0, 255, 255), 3)
        cv.drawContours(dispCorners, corners, -1, (0, 255, 0), 10)

        # Sorting the corners and converting them to desired shape.
        corners = sorted(np.concatenate(corners).tolist())
        
        coords = np.zeros((4,2), dtype=np.int32)
        
        # Put coords in array
        for index, cornerMark in enumerate(corners):
            coords[index, :] = cornerMark

        coords = np.array(corners, dtype=np.float32).reshape(4, 2)
        reCoords = reorder(coords)
        
        dst_width = 960
        dst_height = 540

        dst_pts =  np.array([
            [0,0], #Topleft
            [dst_width -1, 0], #Topright
            [dst_width - 1, dst_height-1], #bottomright
            [0, dst_height -1] #bottomlrft
        ], dtype=np.float32)

        # Calculate the perspective transform matrix
        
        M = cv.getPerspectiveTransform(reCoords, dst_pts)

        # Apply the perspective transform
        warpedImage = cv.warpPerspective(frame, M, (dst_width, dst_height))

        
#Show
    cv.imshow('org', dispCorners)
    cv.imshow('mask', mask)
    cv.imshow("Warped Image", warpedImage)  # Display the warped image



# Wait for the 'Esc' key to break the loop
    if cv.waitKey(24) == 27:
        break


# Release the webcam and close all OpenCV windows
cap.release()
cv.destroyAllWindows()



