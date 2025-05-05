import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2 as cv
import numpy as np
import keyboard

# Laptop with webcam, 3rd camera in hierarchy
cap = cv.VideoCapture(2)

# Single camera in hierarchy
# cap = cv.VideoCapture(0)

detectionFrame = None

low_backgroundRange = np.array([0,0,00])
high_backgroundRange = np.array([179,50,100])

green_low_range = np.array([40,25,100])
green_high_range = np.array([87,255,255])

blue_low_range = np.array([90,25,100])
blue_high_range = np.array([135,255,255])

pink_low_range = np.array([140,25,100])
pink_high_range = np.array([178,255,255])

yellow_low_range = np.array([20,25,100])
yellow_high_range = np.array([40,255,255])

dst_width = 960
dst_height = 1080

dst_pts =  np.array([
    [0,0], #Topleft
    [dst_width -1, 0], #Topright
    [dst_width - 1, dst_height-1], #bottomright
    [0, dst_height -1] #bottomleft
], dtype=np.float32)

# def colourQuant(targetFrame):
#     frameColorQuant = targetFrame
#     reshape = frameColorQuant.reshape((-1,3))
#     reshape = np.float32(reshape)
#     criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_MAX_ITER, 15, 5)
#     K = 5
#     ret,label,center = cv.kmeans(reshape,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

#     center = np.uint8(center)
#     res = center[label.flatten()]
#     res2 = res.reshape((frameColorQuant.shape))
#     return res2

def perspWarp(targetFrame, lowRange, highRange): #returns coordinate array
    # quant = colourQuant(targetFrame)
    hsv = cv.cvtColor(targetFrame,cv.COLOR_BGR2HSV)
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
        return targetFrame, False  # Prevents further errors in calling functions

    else:
        M = cv.getPerspectiveTransform(coords, dst_pts)
        warpedImage = cv.warpPerspective(targetFrame, M, (dst_width, dst_height))
        
        return warpedImage, True
    
def detectObjects(targetFrame,lowRange,highRange):
    # quant = colourQuant(targetFrame)
    hsv = cv.cvtColor(targetFrame,cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv,lowRange,highRange)
    blur = cv.GaussianBlur(mask, (5,5),0)
    _,mask = cv.threshold(blur, 127, 255, cv.THRESH_BINARY)

    all_obj_coordinates = []

    #Contours
    
    blank = np.zeros((dst_height, dst_width, 4), dtype=np.uint8)
    frameContor = blank.copy()
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    if not contours:  # Check if any contours were found
        return np.zeros((dst_height, dst_width, 4), dtype=np.uint8), []

    if contours:
        for indvCnt in contours: #loop throuch individual contours
            if cv.contourArea(indvCnt)>1000: # filters smaller areas less than 1000px out
                epsilon = 0.02 * cv.arcLength(indvCnt, True) 
                approx = cv.approxPolyDP(indvCnt, epsilon, True)

                max_iterations = 10  # Set a maximum number of iterations
                iterations = 0

                while len(approx) != 4 and iterations < max_iterations:
                    epsilon += 0.005 * cv.arcLength(indvCnt, True)
                    approx = cv.approxPolyDP(indvCnt, epsilon, True)
                    iterations += 1

                if len(approx) != 4: # check if approx is 4, if not, skip this contour.
                    print("Could not find 4 corners, skipping contour.")
                    continue

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
        return frameContor, all_obj_coordinates 
    else:
        return np.zeros((dst_height, dst_width, 4), dtype=np.uint8), []
    
def reorder(pts): #returns array
    sum = np.sum(pts,axis=1)

    topLeft = pts[np.argmin(sum)]
    bottomRight = pts[np.argmax(sum)]

    diff = np.diff(pts,axis=1)

    topRight = pts[np.argmin(diff)]
    bottomLeft = pts[np.argmax(diff)]

    return np.array([topLeft,topRight,bottomRight,bottomLeft])

def generate_tscn_multiple_collision_polygons_2d(polygons_coordinates, colour, output_file):
    """
    Generates a Godot .tscn file with multiple CollisionPolygon2D nodes based on a list of coordinate lists.

    Args:
        polygons_coordinates: A list of lists of tuples, where each inner list represents a polygon's vertices.
        output_file: The name of the output .tscn file.
    """

    with open(output_file + ".tscn", "w") as f:
        f.write(f"[gd_scene load_steps=1 format=3 uid=\"uid://multiple_collision_polygons_gen\"]\n\n")
        f.write(f"[node name=\"{output_file}\" type=\"StaticBody2D\"]\n\n")

        for i, coordinates_list in enumerate(polygons_coordinates):
            godot_coordinates = convert_to_godot_format(coordinates_list)
            f.write(f"[node name=\"Polygon2D_{i}\" type=\"Polygon2D\" parent=\".\"]\n")
            f.write(f"color = Color{colour}\n")
            f.write(f"polygon = PackedVector2Array({godot_coordinates})\n\n")

            f.write(f"[node name=\"CollisionPolygon2D_{i}\" type=\"CollisionPolygon2D\" parent=\".\"]\n")
            f.write(f"polygon = PackedVector2Array({godot_coordinates})\n\n")

def convert_to_godot_format(coordinates_list):
    godot_string = ""
    for x,y in coordinates_list:
        godot_string += f"{x}, {y}, "
    if len(coordinates_list) > 0:
        godot_string = godot_string[:-2] #Remove the last comma and space.
    return godot_string

# Running Loop for webcam
while True:
    cv.waitKey(30)
    ret, frame = cap.read()
    if not ret:
        break



    # trigger persp warp on space bar
    if keyboard.is_pressed('f'): 
        
        # detectG, detectB, detectP, detectY, combined_image = None, None, None, None, None

        detectionFrame = frame.copy()
        print("Perspective Warp Triggered")
        warp, warpSuccess = perspWarp(detectionFrame, low_backgroundRange, high_backgroundRange)

        if warpSuccess is True:
            print("Perspective Warp Successful")
            # if key == 102: #F key to detect

            detected_coordinates_blue, detected_coordinates_green, detected_coordinates_pink, detected_coordinates_yellow = [],[],[],[]
            
            detectB, detected_coordinates_blue = detectObjects(warp, blue_low_range, blue_high_range)
            blue_success = bool(detected_coordinates_blue)
            print("Blue Detect: ", blue_success)

            detectG, detected_coordinates_green = detectObjects(warp, green_low_range, green_high_range)
            green_success = bool(detected_coordinates_green)
            print("Green Detect: ", green_success)

            detectP, detected_coordinates_pink = detectObjects(warp, pink_low_range, pink_high_range)
            pink_success = bool(detected_coordinates_pink)
            pink_success = bool(detected_coordinates_pink)
            print("Pink Detect: ", pink_success)

            detectY, detected_coordinates_yellow = detectObjects(warp, yellow_low_range, yellow_high_range)
            yellow_success = bool(detected_coordinates_yellow)
            print("Yellow Detect: ", yellow_success)
            
            # Combine detection contours into one display image
            combined_image = np.zeros((dst_height, dst_width, 4), dtype=np.uint8)

            if blue_success:
                combined_image += detectB
            if green_success:
                combined_image += detectG
            if pink_success:
                combined_image += detectP
            if yellow_success:
                combined_image += detectY
                
            # Clip pixel values to stay within the 0-255 range
            combined_image = np.clip(combined_image, 0, 255)
            combined_image = np.uint8(combined_image)

            if combined_image is not None and warp is not None:
                resized_warp = cv.resize(warp, (warp.shape[1] // 2, warp.shape[0] // 2))
                cv.imshow("Captured Image", resized_warp)
                if combined_image is not None:
                    resized_combined = cv.resize(combined_image, (combined_image.shape[1] // 2, combined_image.shape[0] // 2))
                    cv.imshow('Warp and Detect', resized_combined)

                    # if blue_success:
                    generate_tscn_multiple_collision_polygons_2d(detected_coordinates_blue, (0.2,0.9,1,1),r"SonjaBerg_GameLevelCreator\DetectedPolygons\blue_polygons")
                    # if green_success:
                    generate_tscn_multiple_collision_polygons_2d(detected_coordinates_green, (0.5,1,0.5,1),r"SonjaBerg_GameLevelCreator\DetectedPolygons\green_polygons")
                    # if pink_success:
                    generate_tscn_multiple_collision_polygons_2d(detected_coordinates_pink, (1,0.5,1,1),r"SonjaBerg_GameLevelCreator\DetectedPolygons\pink_polygons")
                    # if yellow_success:
                    generate_tscn_multiple_collision_polygons_2d(detected_coordinates_yellow, (1,1,0.5,1),r"SonjaBerg_GameLevelCreator\DetectedPolygons\yellow_polygons")
                    print("TSCN Files Generated")

        else:
            print("Perspective Warp Failed")
#Show
    cv.imshow('Original Cam Feed', frame)


# Wait for the 'Esc' key to break the loop
    if keyboard.is_pressed('esc'):        
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv.destroyAllWindows()