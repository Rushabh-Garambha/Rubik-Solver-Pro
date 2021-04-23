import cv2 as cv
import numpy as np
from color_calibration import get_color
from cube_display import CubeDisplay

cd = CubeDisplay()

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()


def nothing(*args, **kwargs):
    pass


min_threshold = 74
max_threshold = 142
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
    # border_color = (255,0,255)
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


while True:
    cv.imshow('cube_state', cd.cube_frame)
    ret, frame = cap.read()
    height, width, _ = frame.shape
    img = frame.copy()
    detect_box_size = cv.getTrackbarPos('box_size', 'trackbar')
    start_x = (width - detect_box_size) // 2
    start_y = (height - detect_box_size) // 2
    # frame, face_cubie_coords = create_live_feed_cube_square(frame,start_x,start_y,detect_box_size)
    frame, live_instance_cords = create_live_feed_cube_square(frame, 10, height - 100, 90, (128, 128, 128))

    # pt1, pt2 = face_cubie_coords[0]
    # offset = (pt2[0] - pt1[0]) // 10

    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = img_hsv[:,:,2]
    blurred_img = cv.medianBlur(img_gray, 3)

    sharpen_k = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    filtered_img = cv.filter2D(img, -1, sharpen_k)
    gray_filtered_img = cv.filter2D(blurred_img, -1, sharpen_k)
    #
    # min_threshold = cv.getTrackbarPos('min_threshold', 'trackbar')
    # max_threshold = cv.getTrackbarPos('max_threshold', 'trackbar')

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
                final_contours.append((x, y, w, h,x+(w//2),y+(h//2)))

    # img_cnt = cv.drawContours(img, contours, -1, (0, 255, 0), 3)
    # print(len(final_contours))
    # print(final_contours)
    final_contours.sort(key = lambda x : (x[1]//20, x[0]//20))

    # cv.imshow('contours', img_cnt)

    for fc in final_contours:
        cv.putText(frame, str(final_contours.index(fc)), (fc[0], fc[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv.rectangle(frame, (fc[0] + 2, fc[1] + 2), (fc[0] + fc[2] - 2, fc[1] + fc[3] - 2), (255, 255, 255), 2)

    side = []
    if (len(final_contours) == 9):
        # print("inside if")

        for fc in final_contours:
            # cv.putText(img, str(final_contours.index(fc)), (fc[0], fc[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            # cv.rectangle(img, (fc[0] + 2, fc[1] + 2), (fc[0] + fc[2] - 2, fc[1] + fc[3] - 2), (255, 255, 255), 2)
            # change 9 if not 3*3

            x1, y1 = fc[0] + cubie_offset , fc[1] + cubie_offset
            x2, y2 = fc[0] + fc[2] - cubie_offset , fc[1] + fc[3] - cubie_offset

            color = get_color(img, x1, x2, y1, y2)
            # color = get_color(img_hsv, x1, x2, y1, y2)
            # print(color)

            if color is not None:
                side.append(color)
            else:
                break


        if (len(side) == 9):
            # center = side[4]
            detected_side = side
            # if (center == 'y'):
            #     cube[0] = side
            # elif (center == 'b'):
            #     cube[1] = side
            # elif (center == 'r'):
            #     cube[2] = side
            # elif (center == 'g'):
            #     cube[3] = side
            # elif (center == 'o'):
            #     cube[4] = side
            # elif (center == 'w'):
            #     cube[5] = side
            # print(side)
    # print(detected_side)
    frame = cd.update_colors(detected_side,frame, live_instance_cords,list(range(9)))
    # _, sample_colors = cd.get_random_face_cubie_colors_pair()

    # cv.putText(frame, "Keep your cube inside square for better results.", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5,
    #            (0, 0, 0), 2)

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

    cv.imshow('live_feed', frame)
    #cv.imshow('side2', img)

    # if cv.waitKey(1) == ord('r'):
        # TODO: reset state to try everything again

    # g_blurred = cv.GaussianBlur(img_gray, (kernel_size, kernel_size), 0)
    #

    # g_blurred = cv.GaussianBlur(img_gray, (kernel_size,kernel_size), 0)
    # m_blurred = cv.medianBlur(img_gray,kernel_size)
    # r_blurred = cv.blur(img_gray,(kernel_size,kernel_size))
    #
    # min_canny_threshold = cv.getTrackbarPos('min_canny_threshold', 'trackbar')
    # max_canny_threshold = cv.getTrackbarPos('max_canny_threshold', 'trackbar')

    # g_canny = cv.Canny(g_blurred, min_canny_threshold, max_canny_threshold)
    # m_canny = cv.Canny(m_blurred, min_canny_threshold, max_canny_threshold)
    # r_canny = cv.Canny(r_blurred, min_canny_threshold, max_canny_threshold)
    #
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))
    # g_dilatedFrame = cv.dilate(g_canny, kernel)
    # m_dilatedFrame = cv.dilate(m_canny, kernel)
    # r_dilatedFrame = cv.dilate(r_canny,kernel)
    # canny = np.hstack((g_dilatedFrame,m_dilatedFrame,r_dilatedFrame))
    # cv.imshow('canny_edges',canny)
    #
    # contours, hierarchy = cv.findContours(m_dilatedFrame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # final_contours = []
    # for contour in contours:
    #
    #     perimeter = cv.arcLength(contour, True)
    #     approx = cv.approxPolyDP(contour, 0.1 * perimeter, True)
    #     if len(approx) == 4:
    #         area = cv.contourArea(contour)
    #         (x, y, w, h) = cv.boundingRect(approx)
    #
    #         # Find aspect ratio of boundary rectangle around the countours.
    #         ratio = w / float(h)
    #
    #         # Check if contour is close to a square.
    #         if ratio >= 0.8 and ratio <= 1.2 and w >= 30 and w <= 60 and area / (w * h) > 0.4:
    #             final_contours.append((x, y, w, h))
    #
    # if len(final_contours) < 9:
    #     print('break 63')
    #     continue
    #     #return []
    #
    #
    # found = False
    # contour_neighbors = {}
    # for index, contour in enumerate(final_contours):
    #     (x, y, w, h) = contour
    #     contour_neighbors[index] = []
    #     center_x = x + w / 2
    #     center_y = y + h / 2
    #     radius = 1.5
    #     neighbor_positions = [
    #         # top left
    #         [(center_x - w * radius), (center_y - h * radius)],
    #
    #         # top middle
    #         [center_x, (center_y - h * radius)],
    #
    #         # top right
    #         [(center_x + w * radius), (center_y - h * radius)],
    #
    #         # middle left
    #         [(center_x - w * radius), center_y],
    #
    #         # center
    #         [center_x, center_y],
    #
    #         # middle right
    #         [(center_x + w * radius), center_y],
    #
    #         # bottom left
    #         [(center_x - w * radius), (center_y + h * radius)],
    #
    #         # bottom middle
    #         [center_x, (center_y + h * radius)],
    #
    #         # bottom right
    #         [(center_x + w * radius), (center_y + h * radius)],
    #     ]
    #
    #     for neighbor in final_contours:
    #         (x2, y2, w2, h2) = neighbor
    #         for (x3, y3) in neighbor_positions:
    #             # The neighbor_positions are located in the center of each
    #             # contour instead of top-left corner.
    #             # logic: (top left < center pos) and (bottom right > center pos)
    #             if (x2 < x3 and y2 < y3) and (x2 + w2 > x3 and y2 + h2 > y3):
    #                 contour_neighbors[index].append(neighbor)
    #
    #     # Step 3/4: Now that we know how many neighbors all contours have, we'll
    #     # loop over them and find the contour that has 9 neighbors, which
    #     # includes itself. This is the center piece of the cube. If we come
    #     # across it, then the 'neighbors' are actually all the contours we're
    #     # looking for.
    # for (contour, neighbors) in contour_neighbors.items():
    #     if len(neighbors) == 9:
    #         found = True
    #         final_contours = neighbors
    #         break
    #
    # if not found:
    #     print('break 124')
    #     continue
    #     #return []
    #
    #     # Step 4/4: When we reached this part of the code we found a cube-like
    #     # contour. The code below will sort all the contours on their X and Y
    #     # values from the top-left to the bottom-right.
    #
    #     # Sort contours on the y-value first.
    # y_sorted = sorted(final_contours, key=lambda item: item[1])
    #
    # # Split into 3 rows and sort each row on the x-value.
    # top_row = sorted(y_sorted[0:3], key=lambda item: item[0])
    # middle_row = sorted(y_sorted[3:6], key=lambda item: item[0])
    # bottom_row = sorted(y_sorted[6:9], key=lambda item: item[0])
    #
    # sorted_contours = top_row + middle_row + bottom_row
    # img_cnt = cv.drawContours(img, sorted_contours, -1, (0, 255, 0), 3)
    # #cv.imshow('cntr', img_cnt)
    # #return sorted_contours
    #
    # # sharpening_kernel_2 = np.array([[-1, -1, -1], [-1, 9, -1], a[-1, -1, -1]])
    # # img_median_blur = cv.medianBlur(img_gray, 5)
    # # img_sharpened_2 = cv.filter2D(img_median_blur, -1, sharpening_kernel_2)
    # #
    # # ret, img_binthr = cv.threshold(img_sharpened_2, min_threshold, max_threshold, cv.THRESH_BINARY)
    # # contours, hierarchy = cv.findContours(img_binthr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # # img_cnt = cv.drawContours(img, contours, -1, (0, 255, 0), 3)
    # # cv.imshow('cntr', img_cnt)
    if cv.waitKey(1) == ord('p'):
        break
cv.waitKey(0)
cap.release()
cv.destroyAllWindows()
