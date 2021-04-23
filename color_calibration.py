import numpy as np
import cv2 as cv

def get_color(img_hsv, x1, x2, y1, y2):
    color_code = np.array(cv.mean(img_hsv[y1:y2,x1:x2])).astype(int)
    b, g, r = color_code[0], color_code[1], color_code[2]
    # print(b, g, r)


    if b < 100 and g > 120 and r > 120 and np.abs(g - r) < 30:
        return 'y'
    elif b > 120 and g > 120 and r > 100:
        return 'w'
    elif b > g > r:
        return 'b'
    elif g > b and g > r and np.abs(b - r) < 50:
        return 'g'

    # elif r > b and r > g and np.abs(b - g) < 50 and b < 60:
    elif r > b and r > g and np.abs(b - g) < 30 and b < 80:
        return 'r'
    elif r > g > b and r > 120:
        return 'o'

    # hue, sat, val = 0,0,0
    #
    # for y in range(y2, y1):
    #     for x in range(x2, x1):
    #         h,s,v,_ = img_hsv[y,x]
    #         hue += h
    #         sat += s
    #         val += v
    # area = (y2-y1+1)*(x2-x1+1)
    # print(hue * 2, sat, val, area)
    # hue = hue // area
    # sat = sat // area
    # val = val // area
    # color_hsv_code = np.array(cv.mean(img_hsv[x1:x2,y1:y2])).astype(int)
    # hue, sat, val = color_hsv_code[0],color_hsv_code[1],color_hsv_code[2]
    # print(hue * 2, sat, val)
    # # if(sat < 100): #can add val > 70/80 for all cases for strict bounds
    # #    return 'w'
    # # elif(hue > 60 and hue < 80):
    # if (hue > 60 and hue < 80):
    #     return 'g'
    # elif(hue > 33 and hue < 45):
    #     return 'y'
    # elif(hue > 5 and hue < 15 and val > 80):
    #     return 'o'
    # elif(hue < 5 or hue > 170):
    #     return 'r'
    # elif(hue > 105 and hue < 130):
    #     return 'b'
    # elif (sat < 100):
    #     return 'w'
    return
