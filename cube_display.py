import cv2 as cv
import numpy as np
import random
from rubik_solver import utils

class CubeDisplay():

    def __init__(self):
        self.x_cubies = 9
        self.y_cubies = 12
        self.faces = ['top', 'left', 'front', 'right', 'back', 'bottom']
        self.colors = {}
        self.set_colors()
        self.cube_frame = np.zeros((600, 800, 3), dtype=np.uint8)
        self.cubies_mapping = {}
        img = cv.imread('logo.jpg')
        imh, imw , _ = img.shape
        self.cube_frame[410:410+imh,410:410+imw,:] = img
        # self.cube = [''] * 54
        self.setup()


    def set_colors(self):
        """
        setting color values for present string name
        :return:
        """
        self.colors['y'] = (0, 255, 255)
        self.colors['b'] = (255, 0, 0)
        self.colors['r'] = (0, 0, 255)
        self.colors['g'] = (0, 255, 0)
        self.colors['o'] = (0, 165, 255)
        self.colors['w'] = (255, 255, 255)
        return


    def get_center_color_mapping(self):
        """
        create dictionary mapping for faces and its center cubie's color value
        mapping is as below
        4 (Upper center): YELLOW
        13 (Left center): BLUE
        22 (Front center): RED
        31 (Right center): GREEN
        40 (Back center): ORANGE
        49 (Down center): WHITE
        :return:
        """
        return dict(zip(self.faces, self.colors.keys()))



    def get_cube_display_mappings(self):
        """
        this gives mapping for cubies set per each face and it's indices in drawn display window grid.
        :return:
        """
        #little bit hard coded stuff
        mapping = {}
        # mapping['top'] = [3, 4, 5, 15, 16, 17, 27, 28, 29]
        # mapping['left'] = []
        # mapping['front'] = []
        # mapping['right'] = []
        # mapping['back'] = []
        # mapping['bottom'] = [75, 76, 77, 87, 88, 89, 99, 100, 101]
        mapping['y'] = [3, 4, 5, 15, 16, 17, 27, 28, 29]
        mapping['b'] = []
        mapping['r'] = []
        mapping['g'] = []
        mapping['o'] = []
        mapping['w'] = [75, 76, 77, 87, 88, 89, 99, 100, 101]

        # creating mappings
        for i in range(3, 6):
            for j in range(self.y_cubies):
                id = (i * self.y_cubies) + j
                if (j < 3):
                    mapping['b'].append(id)
                elif (j < 6):
                    mapping['r'].append(id)
                elif (j < 9):
                    mapping['g'].append(id)
                else:
                    mapping['o'].append(id)

        return mapping


    def create_cube_face(self):
        """
        it creates border cube grids to show current cube state.
        :return: cubies_coords
        array of top left and bottom right points tuple for each cubies.(borders excluded)
        """
        height, width, _ = self.cube_frame.shape
        cube_width = width / self.y_cubies
        cube_height = height / self.x_cubies
        cubies_coords = []
        border_color = (128, 128, 128)
        border_thickness = 2
        for i in range(self.x_cubies):
            for j in range(self.y_cubies):
                pt1 = (int(j * cube_width), int(i * cube_height))
                pt2 = (int((j + 1) * cube_width), int((i + 1) * cube_height))
                #just to make that pattern
                if ((i < 3 or i > 5) and j < 6 and j > 2) or (i > 2 and i < 6):
                    cv.rectangle(self.cube_frame, tuple(pt1), tuple(pt2), border_color, border_thickness)

                pt1 = (pt1[0] + border_thickness, pt1[1] + border_thickness)
                pt2 = (pt2[0] - border_thickness, pt2[1] - border_thickness)
                cubies_coords.append([pt1, pt2])
        return cubies_coords


    def color_faces_center_cubies(self):
        """
        it just fills center cubies in each face
        :return:
        """
        for key, values in self.cubies_mapping.items():
            cv.rectangle(self.cube_frame, tuple(self.cubies_coords[values[4]][0]), tuple(self.cubies_coords[values[4]][1]), self.colors[key],
                         -1)
        return

    def solve(self, cube):

        # print(self.cube)

        cube_str = ''
        for side in cube:
            cube_str += ''.join(side)
        #
        # #TODO: check if all the sides are detected correctly
        # print(cube_str)
        try:
            ins = utils.solve(cube_str, 'Kociemba')
            # ins = utils.solve('yrboygrgbowwbbwrwygoogrrbyowyrygywrrybgoowyobobgrwgwbg', 'Kociemba')
            return str(ins)
        except:
            print("something went wrong!")
        return ''



    def update_colors(self,cubies_colors,img = None, cubies_coords = None,cubies_mapping = 0):
        """
        this function fills each face with cubie colors provided.
        validation included  to check color for each cubie is included
        :param cubies_colors: array of 9 color values which will be used to fill cubies
        :param img: image on which it is to be drawn
        :param cubies_coords: coordinates where it should be drawn
        :param cubies_mapping: cubie mapping to get cubie_coords index
        :return:
        """
        if (len(cubies_colors) != 9):
            # print('not enough cubies to update! cubie count', len(cubies_colors))
            return img

        if img is None:
            img = self.cube_frame
            cubies_coords = self.cubies_coords
            cubies_mapping = self.cubies_mapping[cubies_colors[4]]
            print('updating Cube state')

        for c_i in range(len(cubies_colors)):
            cv.rectangle(img, tuple(cubies_coords[cubies_mapping[c_i]][0]),
                         tuple(cubies_coords[cubies_mapping[c_i]][1]),self.colors[cubies_colors[c_i]], -1)

        return img

    def get_random_face_cubie_colors_pair(self):
        """
        generates random values for face and array of color value to test the functionality of update_colors
        :return:
        """
        return random.choice(self.faces), [random.choice(list(self.colors.values())) for i in range(9)]

    def validate_face_center_color(self,face, face_cubies_colors):
        """
        validates whether given center cubies' color value is what it supposed to be from face_center_colors_mapping
        :param face: which side of cube
        :param face_cubies_colors: list of color values for each cubies
        :return:
        """
        return self.face_center_colors[face] == face_cubies_colors[4]

    def setup(self):
        """
        this function does the creating and setting required class attributes which are to be used for display.
        :return:
        """
        self.face_center_colors = self.get_center_color_mapping()
        self.cubies_mapping = self.get_cube_display_mappings()
        self.cubies_coords = self.create_cube_face()
        self.color_faces_center_cubies()

    def display(self):
        self.setup()
        while True:
            face, face_cubies_colors = self.get_random_face_cubie_colors_pair()
            face_update_flag = self.validate_face_center_color(face, face_cubies_colors)
            if face_update_flag:
                self.update_colors(face_cubies_colors, self.cube_frame,self.cubies_coords,self.cubies_mapping[face])
            cv.imshow('cube', self.cube_frame)
            if cv.waitKey(1) == ord('q'):
                break
        return

#cd.display()
#
