import cv2 as cv
import numpy as np
from color_calibration import get_color
from cube_display import CubeDisplay

cd = CubeDisplay()

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()


min_thr = 74
max_thr = 142
cv.namedWindow('live_feed')

kernel_size = 3
cube = np.empty((6,9), dtype='str')
detected_side = []
solution = ''
cubie_offset = 2

def create_live_feed_cube_square(img, start_x, start_y, box_size, border_color=(255, 0, 255), cube_size=3, ):
    box_size = max(90, box_size)
    cubie_s = box_size / cube_size
    cubies_coords = []
    border_thickness = 1
    for i in range(cube_size):
        for j in range(cube_size):
            pt1 = (start_x + int(j * cubie_s), start_y + int(i * cubie_s))
            pt2 = (start_x + int((j + 1) * cubie_s), start_y + int((i + 1) * cubie_s))

            cv.rectangle(img, tuple(pt1), tuple(pt2), border_color, border_thickness)

            pt1 = (pt1[0] + border_thickness, pt1[1] + border_thickness)
            pt2 = (pt2[0] - border_thickness, pt2[1] - border_thickness)
            cubies_coords.append([pt1, pt2])
    return img, cubies_coords

def detect_contours(img,min_threshold,max_threshold):

    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = img_hsv[:, :, 2]
    blurred_img = cv.medianBlur(img_gray, 3)

    sharpen_k = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    gray_filtered_img = cv.filter2D(blurred_img, -1, sharpen_k)

    ret, thresholded_img = cv.threshold(gray_filtered_img, min_threshold, max_threshold, cv.THRESH_BINARY)

    morph_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    eroded_img = cv.erode(thresholded_img, morph_kernel, iterations=1)
    dilate_img = cv.dilate(eroded_img, morph_kernel, iterations=1)

    contours, hierarchy = cv.findContours(dilate_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    final_contours = []
    for contour in contours:
        perimeter = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.1 * perimeter, True)
        if len(approx) == 4:
            area = cv.contourArea(contour)
            (x, y, w, h) = cv.boundingRect(approx)

            # Find aspect ratio of boundary rectangle around the countours.
            ratio = w / float(h)

            # Check if contour is close to a square.
            if ratio >= 0.8 and ratio <= 1.2 and w >= 30 and w <= 60 and area / (w * h) > 0.4:
                final_contours.append((x, y, w, h, x + (w // 2), y + (h // 2)))

    # img_cnt = cv.drawContours(img, contours, -1, (0, 255, 0), 3)
    # print(final_contours)
    final_contours.sort(key=lambda x: (x[1] // 20, x[0] // 20))

    return final_contours

while True:
    # Display Cube State
    cv.imshow('cube_state', cd.cube_frame)

    # Initial Frame Processing
    ret, frame = cap.read()
    height, width, _ = frame.shape
    img = frame.copy()

    # Draw instace box in frame
    frame, live_instance_cords = create_live_feed_cube_square(frame, 10, height - 100, 90, (128, 128, 128))

    # Detect contours
    final_contours = detect_contours(img,min_thr,max_thr)

    #display contours
    for fc in final_contours:
        cv.putText(frame, str(final_contours.index(fc)), (fc[0], fc[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv.rectangle(frame, (fc[0] + 2, fc[1] + 2), (fc[0] + fc[2] - 2, fc[1] + fc[3] - 2), (255, 255, 255), 2)

    #color Extraction
    side = []
    if (len(final_contours) == 9):
        for fc in final_contours:
            x1, y1 = fc[0] + cubie_offset , fc[1] + cubie_offset
            x2, y2 = fc[0] + fc[2] - cubie_offset , fc[1] + fc[3] - cubie_offset

            color = get_color(img, x1, x2, y1, y2)

            if color is not None:
                side.append(color)
            else:
                break

        if (len(side) == 9):
            detected_side = side

    #update instance box
    frame = cd.update_colors(detected_side,frame, live_instance_cords,list(range(9)))

    #append state
    if cv.waitKey(1) == ord('a'):
        cd.update_colors(detected_side)
        if len(detected_side) == 9:
            print(detected_side)
            center = detected_side[4]
            if center == 'y':
                cube[0] = detected_side
            elif center == 'b':
                cube[1] = detected_side
            elif center == 'r':
                cube[2] = detected_side
            elif center == 'g':
                cube[3] = detected_side
            elif center == 'o':
                cube[4] = detected_side
            elif center == 'w':
                cube[5] = detected_side


    #solve cube state
    if cv.waitKey(1) == ord('s'):
        # TODO: add condition to check if all the sides are detected
        print(cube)
        ins_str = cd.solve(cube)
        print("test : " + ins_str)
        if len(ins_str) != 0:
            solution = "Solution: " + ins_str[1:-1]
        # also once we have solution we won't need to show the contours..
        # we will still need to detect them to show arrows,
        # but for that we will need to get the sequence correct --hint for self-- do it with centers rather than corners


    frame = cv.putText(frame, solution, (30,30), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 2)

    #Display Live Feed
    cv.imshow('live_feed', frame)

    # if cv.waitKey(1) == ord('r'):
        # TODO: reset state to try everything again

    if cv.waitKey(1) == ord('p'):
        break
cv.waitKey(0)
cap.release()
cv.destroyAllWindows()
