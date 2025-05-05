import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2 as cv
import numpy as np

# Turn on Laptop's webcam
cap = cv.VideoCapture(0)
# cap = cv.VideoCapture(1) #Laptop back camera

#All colours except black/white and super desaturated
#low_colourRange = np.array([0,150,20])
#high_colourRange = np.array([179,255,255])

detectionFrame = None

low_whiteRange = np.array([0,0,100])
high_whiteRange = np.array([179,50,255])

green_low_range = np.array([40,25,20])
green_high_range = np.array([87,255,255])

red1_low_range = np.array([160,50,20])
red1_high_range = np.array([179,255,255])

red2_low_range = np.array([0,50,20])
red2_high_range = np.array([10,255,255])

blue_low_range = np.array([90,25,20])
blue_high_range = np.array([135,255,255])

dst_width = 960
dst_height = 540

dst_pts =  np.array([
    [0,0], #Topleft
    [dst_width -1, 0], #Topright
    [dst_width - 1, dst_height-1], #bottomright
    [0, dst_height -1] #bottomleft
], dtype=np.float32)

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

    print("colour quant complete")

    return res2

def reorder(pts): #returns array
    sum = np.sum(pts,axis=1)

    topLeft = pts[np.argmin(sum)]
    bottomRight = pts[np.argmax(sum)]

    diff = np.diff(pts,axis=1)

    topRight = pts[np.argmin(diff)]
    bottomLeft = pts[np.argmax(diff)]

    print("reorder complete")

    return np.array([topLeft,topRight,bottomRight,bottomLeft])

def perspWarp(targetFrame, lowRange, highRange): #returns coordinate array
    quant = colourQuant(targetFrame)
    hsv = cv.cvtColor(quant,cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv,lowRange,highRange)

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

    if np.all(coords == 0):  # If no valid points were found
        print("No valid contours detected for perspective warp.")
        return None  # Prevents further errors in calling functions

    M = cv.getPerspectiveTransform(coords, dst_pts)
    warpedImage = cv.warpPerspective(targetFrame, M, (dst_width, dst_height))

    print("perspecitve warp complete")
    
    return warpedImage
    
def detectObjects(targetFrame,lowRange,highRange):
    quant = colourQuant(targetFrame)
    hsv = cv.cvtColor(quant,cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv,lowRange,highRange)
    blur = cv.GaussianBlur(mask, (5,5),0)
    _,mask = cv.threshold(blur, 127, 255, cv.THRESH_BINARY)

    all_obj_coordinates = []

    #Contours
    
    blank = np.zeros((dst_height, dst_width, 4), dtype=np.uint8)
    frameContor = blank.copy()
    contours, _ = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    if contours is not None:
        for indvCnt in contours: #loop throuch individual contours
            if cv.contourArea(indvCnt)>1000: # filters smaller areas less than 1000px out
                epsilon = 0.02 * cv.arcLength(indvCnt, True) 
                approx = cv.approxPolyDP(indvCnt, epsilon, True)

                while len(approx) != 4:
                    epsilon += 0.005 * cv.arcLength(indvCnt, True)
                    approx = cv.approxPolyDP(indvCnt, epsilon, True)

                if len(approx) == 4: 
                    cv.drawContours(frameContor,[approx],-1,(0,255,255),2)

                    idv_obj_coordinates = []
                    for corner in approx:
                        x, y = corner[0]
                        cv.circle(frameContor, (x, y), 5, (0, 0, 255), -1)
                        cv.putText(frameContor, f"({x},{y})", (x + 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                        idv_obj_coordinates.append((x, y))
                    idv_obj_coordinates = np.array(idv_obj_coordinates, dtype=np.float32).reshape(4, 2)
                    idv_obj_coordinates = reorder(idv_obj_coordinates)
                    all_obj_coordinates.append(idv_obj_coordinates)

        print("detect objects complete")
        return frameContor, all_obj_coordinates 
    else:
        return 0, 0

warp = None

# Running Loop for webcam
while True:
    key = cv.waitKey(30) & 0xFF
    ret, frame = cap.read()
    if not ret:
        break

    # trigger persp warp on space bar
    if key == 32: 
        
        detectionFrame = frame.copy()
        combined_image = None
        print("Perspective Warp Triggered")
        warp = perspWarp(detectionFrame, low_whiteRange, high_whiteRange)
        print("Perspective Warp Successful")
    if warp is not None:
        if key == 102: #F key to detect
            detectG, detectR1, detectR2, detectB, combined_image = None, None, None, None, None

            detectR1, detected_coordinates_red1 = detectObjects(warp, red1_low_range, red1_high_range)
            print("red1 success")
            detectR2, detected_coordinates_red2 = detectObjects(warp, red2_low_range, red2_high_range)
            print("red2 success")
            detectB, detected_coordinates_blue = detectObjects(warp, blue_low_range, blue_high_range)
            print("blue success")
            detectG, detected_coordinates_green = detectObjects(warp, green_low_range, green_high_range)
            print("green success")
            
            if detectG is not None and detectR1 is not None and detectR2 is not None and detectB is not None:
                combined_image = detectG + detectR1 + detectR2 + detectB
                # Clip pixel values to stay within the 0-255 range
                combined_image = np.clip(combined_image, 0, 255)
                combined_image = np.uint8(combined_image)
                print("Detection Successful")

    if detectionFrame is not None and warp is not None:
        cv.imshow("Captured Image", warp)
        if combined_image is not None:
            cv.imshow('Warp and Detect', combined_image)

     
#Show
    cv.imshow('Original Cam Feed', frame)

# Wait for the 'Esc' key to break the loop
    if key == 27:        
        # generate_tscn_multiple_collision_polygons_2d(detected_coordinates_red1, "multiple_polygons.tscn")
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv.destroyAllWindows()


