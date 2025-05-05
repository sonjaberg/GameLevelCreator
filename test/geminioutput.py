import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

# Define color range for object detection (adjust these values)
# For white/grey:
low_range = np.array([0, 0, 70])
high_range = np.array([179, 50, 255])

# For other colors (example):
# low_range = np.array([0, 150, 20])  # Adjust these
# high_range = np.array([179, 255, 255]) # Adjust these

warpComplete = False
perspCoords = None  # Initialize perspCoords

def reorder(pts):
    """Reorders corner points to topLeft, topRight, bottomRight, bottomLeft."""
    pts = pts.reshape((4, 2))  # Ensure correct shape
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    topLeft = pts[np.argmin(s)]
    bottomRight = pts[np.argmax(s)]
    topRight = pts[np.argmin(diff)]
    bottomLeft = pts[np.argmax(diff)]
    return np.array([topLeft, topRight, bottomRight, bottomLeft], dtype=np.float32)  # Return float32

def perspWarp(frame, hsv):
    """Performs perspective warp on the detected object."""
    mask = cv.inRange(hsv, low_range, high_range)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) # Use RETR_EXTERNAL and CHAIN_APPROX_SIMPLE

    largest_area = 0
    target_contour = None
    corners = None

    for contour in contours:
        area = cv.contourArea(contour)
        if area > largest_area:
            largest_area = area
            ep = 0.02 * cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, ep, True)
            if len(approx) == 4:
                target_contour = contour
                corners = approx.reshape(4, 2)  # Reshape corners
    
    if corners is not None:
        corners = reorder(corners) # Reorder corners
        return corners
    else:
        return None  # Return None if no suitable contour is found

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frameColorQuant = frame.copy()
    reshape = frameColorQuant.reshape((-1, 3)).astype(np.float32) # Directly convert to float32
    criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_MAX_ITER, 10, 1.0) # Reduced iterations and epsilon
    K = 8 # Reduced K
    _, label, center = cv.kmeans(reshape, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()].reshape(frameColorQuant.shape)

    hsv = cv.cvtColor(res, cv.COLOR_BGR2HSV)

    if cv.waitKey(1) == 32:  # Spacebar to trigger warp
        perspCoords = perspWarp(frame, hsv)
        if perspCoords is not None:
            warpComplete = True
        else:
            print("No suitable object found for warping.") # Inform user if no object found

    dst_width = 960
    dst_height = 540
    dst_pts = np.array([[0, 0], [dst_width - 1, 0], [dst_width - 1, dst_height - 1], [0, dst_height - 1]], dtype=np.float32)

    warpedImage = frame.copy() # Initialize warpedImage with original frame

    if warpComplete and perspCoords is not None:  # Check if warpComplete and perspCoords are valid
        M = cv.getPerspectiveTransform(perspCoords, dst_pts)
        warpedImage = cv.warpPerspective(frame, M, (dst_width, dst_height))

    cv.imshow('Original', frame)
    cv.imshow("Warped Image", warpedImage)

    if cv.waitKey(1) == 27:  # Escape key to exit
        break

cap.release()
cv.destroyAllWindows()