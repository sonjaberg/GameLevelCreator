import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2 as cv
import numpy as np

# Turn on Laptop's webcam
cap = cv.VideoCapture(0)

#All colours except black/white and super desaturated
#low_colourRange = np.array([0,150,20])
#high_colourRange = np.array([179,255,255])

warpComplete = False

detectionFrame = None

low_whiteRange = np.array([0,0,70])
high_whiteRange = np.array([179,50,255])

obj_low_range = np.array([0,150,20])
obj_high_range = np.array([179,255,255])

dst_width = 960
dst_height = 540

dst_pts =  np.array([
    [0,0], #Topleft
    [dst_width -1, 0], #Topright
    [dst_width - 1, dst_height-1], #bottomright
    [0, dst_height -1] #bottomlrft
], dtype=np.float32)

def reorder(pts): #returns array
    sum = np.sum(pts,axis=1)

    topLeft = pts[np.argmin(sum)]
    bottomRight = pts[np.argmax(sum)]

    diff = np.diff(pts,axis=1)

    topRight = pts[np.argmin(diff)]
    bottomLeft = pts[np.argmax(diff)]

    return np.array([topLeft,topRight,bottomRight,bottomLeft])

def perspWarp(targetFrame, lowRange, highRange): #returns coordinate array
    quant = colourQuant(targetFrame)
    hsv = cv.cvtColor(quant,cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv,lowRange,highRange)

    #Contours
    contours, _ = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    area = sorted(contours, key=cv.contourArea, reverse=True)[:1]
    
    #Corner detection
    coords = np.zeros((4, 2), dtype=np.float32)
    for c in area:
        if cv.contourArea(c) > 25000:
            ep = 0.02 * cv.arcLength(c, True)
            corners = cv.approxPolyDP(c, ep, True)
            if len(corners) == 4:
                corners = np.concatenate(corners).tolist()
                corners = np.array(corners, dtype=np.float32).reshape(4, 2)
                coords = reorder(corners) #topLeft,topRight,bottomRight,bottomLeft
                break

    warpedImage = None
    if coords is not None:
        M = cv.getPerspectiveTransform(coords, dst_pts)
        warpedImage = cv.warpPerspective(targetFrame, M, (dst_width, dst_height))
        return warpedImage
    else:
        print("No area found")
        return None
    
def detectObjects(targetFrame,lowRange,highRange):
    quant = colourQuant(targetFrame)
    hsv = cv.cvtColor(quant,cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv,lowRange,highRange)
    blur = cv.GaussianBlur(mask, (5,5),0)
    _,mask = cv.threshold(blur, 127, 255, cv.THRESH_BINARY)

    #Contours
    frameContor = targetFrame.copy()
    contours, _ = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    for indvCnt in contours: #loop throuch individual contours
         if cv.contourArea(indvCnt)>1000: # filters smaller areas less than 1000px out
            epsilon = 0.02 * cv.arcLength(indvCnt, True)  # Adjust epsilon as needed
            approx = cv.approxPolyDP(indvCnt, epsilon, True)  # Complete approxPolyDP
            cv.drawContours(frameContor,[approx],-1,(0,255,255),2)
            for corner in approx:
                x, y = corner[0]
                cv.circle(frameContor, (x, y), 5, (0, 0, 255), -1)
    return frameContor 
    
def colourQuant(targetFrame):
        frameColorQuant = targetFrame
        reshape = frameColorQuant.reshape((-1,3))
        reshape = np.float32(reshape)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_MAX_ITER, 15, 5)
        K = 10
        ret,label,center = cv.kmeans(reshape,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((frameColorQuant.shape))
        return res2

while True:
    ret, frame = cap.read()
    if not ret:
        break
   

    # trigger persp warp on space bar
    if cv.waitKey(24) == 32:
        detectionFrame = frame.copy()
        warp = perspWarp(detectionFrame, low_whiteRange, high_whiteRange)
        if warp is not None:
            detect = detectObjects(warp, obj_low_range, obj_high_range)
            warpComplete = True 

    # warpedImage = None

    # if warpComplete == True:
    #     # Calculate the perspective transform matrix
    #     M = cv.getPerspectiveTransform(perspCoords, dst_pts)

    #     # Apply the perspective transform
    #     warpedImage = cv.warpPerspective(frame, M, (dst_width, dst_height))


    # detect = frame.copy()
    
    # detect = detectObjects(warpedImage,obj_low_range,obj_high_range)

    

        
#Show
    cv.imshow('org', frame)

    if warpComplete and detectionFrame is not None and detect is not None:
        cv.imshow("Warped Image", detectionFrame)
        cv.imshow('obj', detect)
    # cv.imshow('det', detectFrm)



# Wait for the 'Esc' key to break the loop
    if cv.waitKey(24) == 27:
        break


# Release the webcam and close all OpenCV windows
cap.release()
cv.destroyAllWindows()


