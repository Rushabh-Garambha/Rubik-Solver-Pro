import cv2 as cv
import numpy as np
from color_calibration import get_color
from cube_display import CubeDisplay
from solver import get_steps, update_cube

cd = CubeDisplay()

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

min_thr = 74
max_thr = 142
cv.namedWindow('live_feed')

kernel_size = 3
cube = np.empty((6, 9), dtype='str')
cube_next = np.empty((6, 9), dtype='str')
color_index = ['y', 'b', 'r', 'g', 'o', 'w']
detected_side = []
cur_side = None
next_side = None
moves = None
solved = False
# solution = ''
cubie_offset = 2


#  This function fills cube with sample state so that
#  we don't have to spend time in detecting cube every time
#  during development.
def fill_cube_with_sample_state():
    a = np.array([['y', 'w', 'y', 'w', 'y', 'w', 'y', 'w', 'y'],
                  ['b', 'g', 'b', 'g', 'b', 'g', 'b', 'g', 'b'],
                  ['r', 'o', 'r', 'o', 'r', 'o', 'r', 'o', 'r'],
                  ['g', 'b', 'g', 'b', 'g', 'b', 'g', 'b', 'g'],
                  ['o', 'r', 'o', 'r', 'o', 'r', 'o', 'r', 'o'],
                  ['w', 'y', 'w', 'y', 'w', 'y', 'w', 'y', 'w']])
    return a


def create_live_feed_cube_square(img, start_x, start_y, box_size, border_color=(255, 0, 255), cube_size=3):
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
    # Preview Text with shadow effect
    cv.putText(img, "Preview", (start_x + 1, start_y - 8), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv.putText(img, "Preview", (start_x - 1, start_y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    return img, cubies_coords


def detect_contours(img, min_threshold, max_threshold):
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
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

    final_contours.sort(key=lambda x: (x[1] // 20, x[0] // 20))

    return final_contours


def translate_moves(moves):
    translated = []
    print("Before Translation: {}".format(moves))
    for move in moves:
        if move[0] == 'B':
            m = move.replace("B", "R")
        elif move[0] == 'F':
            m = move.replace("F", "L")
        elif move[0] == 'R':
            m = move.replace("R", "F")
        elif move[0] == 'L':
            m = move.replace("L", "B")
        else:
            m = move
        translated.append(m)
    print("After Translation: {}".format(translated))
    return translated


def draw_arrow(start, end, points, frame):
    x1 = points[start][4]
    y1 = points[start][5]
    x2 = points[end][4]
    y2 = points[end][5]
    return cv.arrowedLine(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)


def show_step(move, final_contours, f):
    if move == 'L':
        f = draw_arrow(0, 6, final_contours, f)
    elif move == "L'":
        f = draw_arrow(6, 0, final_contours, f)
    elif move == 'R':
        f = draw_arrow(8, 2, final_contours, f)
    elif move == "R'":
        f = draw_arrow(2, 8, final_contours, f)
    elif move == 'U':
        f = draw_arrow(2, 0, final_contours, f)
    elif move == "U'":
        f = draw_arrow(0, 2, final_contours, f)
    elif move == 'D':
        f = draw_arrow(6, 8, final_contours, f)
    elif move == "D'":
        f = draw_arrow(8, 6, final_contours, f)
    elif move == 'F':
        f = draw_arrow(1, 5, final_contours, f)
        f = draw_arrow(5, 7, final_contours, f)
        f = draw_arrow(7, 3, final_contours, f)
        f = draw_arrow(3, 1, final_contours, f)
    elif move == "F'":
        f = draw_arrow(5, 1, final_contours, f)
        f = draw_arrow(7, 5, final_contours, f)
        f = draw_arrow(3, 7, final_contours, f)
        f = draw_arrow(1, 3, final_contours, f)
    else:
        # Back or Special case
        f = draw_arrow(2, 0, final_contours, f)
        f = draw_arrow(5, 3, final_contours, f)
        f = draw_arrow(8, 6, final_contours, f)

    return f


def match_sides(side1, side2):
    for i in range(9):
        if side1[i] != side2[i]:
            return False
    return True


# add guide info
append_text = """Press 'a': To add preview to cube"""
append_text_org = (420, 30)
solve_text = "Press 's': To solve the cube"
solve_text_org = (420, 70)
exit_text = "Press 'x': Exit"
exit_text_org = (420, 110)
cd_font_scale = 0.7

cv.putText(cd.cube_frame, append_text, append_text_org, cv.FONT_HERSHEY_SIMPLEX, cd_font_scale, (255, 255, 255), 2)
cv.putText(cd.cube_frame, solve_text, solve_text_org, cv.FONT_HERSHEY_SIMPLEX, cd_font_scale, (255, 255, 255), 2)
cv.putText(cd.cube_frame, exit_text, exit_text_org, cv.FONT_HERSHEY_SIMPLEX, cd_font_scale, (255, 255, 255), 2)

(csw, csh), _ = cv.getTextSize('Cube_solved!', cv.FONT_HERSHEY_SIMPLEX, cd_font_scale, 2)

while True:
    # Display Cube State
    cv.imshow('cube_state', cd.cube_frame)
    # Initial Frame Processing
    ret, frame = cap.read()
    height, width, _ = frame.shape
    img = frame.copy()

    if cv.waitKey(1) == ord('x'):
        break

    if solved:
        cv.putText(frame, "Cube Solved!", ((width - csw) // 2, 30), cv.FONT_HERSHEY_SIMPLEX, cd_font_scale, (255, 0, 0),
                   2)
        cv.imshow('live_feed', frame)
        continue

    # Draw instace box in frame
    frame, live_instance_cords = create_live_feed_cube_square(frame, 10, height - 100, 90, (128, 128, 128))

    # Detect contours
    final_contours = detect_contours(img, min_thr, max_thr)

    # display contours
    for fc in final_contours:
        cv.putText(frame, str(final_contours.index(fc)), (fc[0], fc[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv.rectangle(frame, (fc[0] + 2, fc[1] + 2), (fc[0] + fc[2] - 2, fc[1] + fc[3] - 2), (255, 255, 255), 2)

    # color Extraction
    side = []
    if len(final_contours) == 9:
        for fc in final_contours:
            x1, y1 = fc[0] + cubie_offset, fc[1] + cubie_offset
            x2, y2 = fc[0] + fc[2] - cubie_offset, fc[1] + fc[3] - cubie_offset

            color = get_color(img, x1, x2, y1, y2)

            if color is not None:
                side.append(color)
            else:
                break

        if len(side) == 9:
            detected_side = side

    # update instance box
    frame = cd.update_colors(detected_side, frame, live_instance_cords, list(range(9)))

    # append state
    if cv.waitKey(1) == ord('a'):
        cd.update_colors(detected_side)
        if len(detected_side) == 9:
            print(detected_side)
            center = detected_side[4]
            cube[color_index.index(center)] = detected_side

    # solve cube state
    if cv.waitKey(1) == ord('s'):
        print(cube)
        ins_str = cd.solve(cube)
        print("solution : " + ins_str)
        if len(ins_str) != 0:
            # removing append/solve button info
            cv.putText(cd.cube_frame, append_text, append_text_org, cv.FONT_HERSHEY_SIMPLEX, cd_font_scale, (0, 0, 0),
                       2)
            cv.putText(cd.cube_frame, solve_text, solve_text_org, cv.FONT_HERSHEY_SIMPLEX, cd_font_scale, (0, 0, 0), 2)
            cv.putText(cd.cube_frame, exit_text, exit_text_org, cv.FONT_HERSHEY_SIMPLEX, cd_font_scale, (0, 0, 0), 2)
            cv.putText(cd.cube_frame, exit_text, append_text_org, cv.FONT_HERSHEY_SIMPLEX, cd_font_scale,
                       (255, 255, 255), 2)

            solution = "Solution: " + ins_str[1:-1]
            moves = get_steps(ins_str)
            print("moves : {}".format(moves))
            cur_side = 2
            first_move = moves[0]
            if first_move[0] == 'B':
                next_side = 3
                cube_next = cube.copy()
            else:
                next_side = 2
                cube_next = update_cube(cube, 'r', first_move[0], (len(first_move) == 2))
                if match_sides(cube[cur_side], cube_next[next_side]):
                    if cur_side == 4:
                        next_side = 1
                    else:
                        next_side = cur_side + 1
                    cube_next = cube.copy()
                    first_move = 'S'
                    moves.insert(0, first_move)
            print(first_move, cur_side, next_side)
        # also once we have solution we won't need to show the contours..
        # we will still need to detect them to show arrows,
        # but for that we will need to get the sequence correct --hint for self-- do it with centers rather than corners

    # showing steps if solving the cube
    if moves is not None and len(side) == 9 and not solved:
        if np.all(np.asarray(side) == cube[cur_side]):
            move = moves[0]
            frame = show_step(move, final_contours, frame)

        elif np.all(np.asarray(side) == cube_next[next_side]):
            move = moves[0]
            print("Detected next side")
            if move == 'S' or move[0] == 'B':
                moves = translate_moves(moves) if move[0] == 'B' else translate_moves(moves[1:])
                cur_side = next_side
                move = moves[0]
                if move[0] == 'B':
                    if cur_side == 4:
                        next_side = 1
                    else:
                        next_side = cur_side + 1
                else:
                    cube_next = update_cube(cube, color_index[next_side], move[0], (len(move) == 2))
                print(move, cur_side, next_side)
            else:
                moves = moves[1:]
                if len(moves) == 0:
                    solved = True
                    cube = cube_next.copy()
                    for side in cube:
                        cd.update_colors(side)
                    print("The cube is solved.")
                else:
                    move = moves[0]
                    cube = cube_next.copy()
                    for side in cube:
                        cd.update_colors(side)
                    if move[0] == 'B':
                        if cur_side == 4:
                            next_side = 1
                        else:
                            next_side = cur_side + 1
                    else:
                        next_side = cur_side
                        cube_next = update_cube(cube, color_index[next_side], move[0], (len(move) == 2))
            if not solved and match_sides(cube[cur_side], cube_next[next_side]):
                if cur_side == 4:
                    next_side = 1
                else:
                    next_side = cur_side + 1
                cube_next = cube.copy()
                move = 'S'
                moves.insert(0, move)
            if not solved:
                frame = show_step(move, final_contours, frame)
                print(move, cur_side, next_side)

    # Display Live Feed
    cv.imshow('live_feed', frame)

    if cv.waitKey(1) == ord('r'):
        cube = fill_cube_with_sample_state()
        for i in cube:
            cd.update_colors(i)

cv.waitKey(0)
cap.release()
cv.destroyAllWindows()
