def get_steps(move_string):
    step_list = []

    l = list(move_string[1:-1].split(', '))

    for step in l:
        if len(step) == 2 and step[1] == '2':
            step_list.extend([step[0]] * 2)
        else:
            step_list.append(step)

    return step_list


def update_cube(cube, color, step, isPrime):
    
    if step == 'U':

        s1, i1 = 1, [0, 1, 2]
        s2, i2 = 2, [0, 1, 2]
        s3, i3 = 3, [0, 1, 2]
        s4, i4 = 4, [0, 1, 2]
        f1 = 0

    elif step == 'D':

        s1, i1 = 4, [6, 7, 8]
        s2, i2 = 3, [6, 7, 8]
        s3, i3 = 2, [6, 7, 8]
        s4, i4 = 1, [6, 7, 8]
        f1 = 5

    else:

        # convertig an F move to R move of its left face
        if step == 'F':

            step = 'R'
            if color == 'b':
                color = 'o'
            elif color == 'r':
                color = 'b'
            elif color == 'g':
                color = 'r'
            elif color == 'o':
                color = 'g'
            else:
                print("ERROR: wrong center color")

        if step == 'L':

            s2, s4 = 0, 5
            i1 = [0, 3, 6]
            i3 = [8, 5, 2]

            if color == 'b':
                s1, s3, f1 = 1, 3, 4
                i2 = [2, 1, 0]
                i4 = [6, 7, 8]
            elif color == 'r':
                s1, s3, f1 = 2, 4, 1
                i2 = [0, 3, 6]
                i4 = [0, 3, 6]
            elif color == 'g':
                s1, s3, f1 = 3, 1, 2
                i2 = [6, 7, 8]
                i4 = [2, 1, 0]
            elif color == 'o':
                s1, s3, f1 = 4, 2, 3
                i2 = [8, 5, 2]
                i4 = [8, 5, 2]
            else:
                print("ERROR: wrong center color")

        elif step == 'R':

            s2, s4 = 5, 0
            i1 = [2, 5, 8]
            i3 = [6, 3, 0]

            if color == 'b':
                s1, s3, f1 = 1, 3, 2
                i2 = [0, 1, 2]
                i4 = [8, 7, 6]
            elif color == 'r':
                s1, s3, f1 = 2, 4, 3
                i2 = [2, 5, 8]
                i4 = [2, 5, 8]
            elif color == 'g':
                s1, s3, f1 = 3, 1, 4
                i2 = [8, 7, 6]
                i4 = [0, 1, 2]
            elif color == 'o':
                s1, s3, f1 = 4, 2, 1
                i2 = [6, 3, 0]
                i4 = [6, 3, 0]
            else:
                print("ERROR: wrong center color")
        else:
            print("ERROR: wrong step")

    # final block
    if not isPrime:

        # updating the side rotations
        temp = cube[s1][i1]
        cube[s1][i1] = cube[s2][i2]
        cube[s2][i2] = cube[s3][i3]
        cube[s3][i3] = cube[s4][i4]
        cube[s4][i4] = temp

        # updating the face rotation
        cube[f1] = cube[f1][6, 3, 0, 7, 4, 1, 8, 5, 2]

    else:

        # updating the side rotations
        temp = cube[s4][i4]
        cube[s4][i4] = cube[s3][i3]
        cube[s3][i3] = cube[s2][i2]
        cube[s2][i2] = cube[s1][i1]
        cube[s1][i1] = temp

        # updating the face rotation
        cube[f1] = cube[f1][2, 5, 8, 1, 4, 7, 0, 3, 6]

    return cube
