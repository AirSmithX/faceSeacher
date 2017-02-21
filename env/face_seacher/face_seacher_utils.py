__author__ = 'Air'
import cv2
import numpy as np
import random

def connection_area(_img, _seed):
    img = _img.copy()
    img_re, rect = cv2.floodFill(img, None, _seed, (255, 255, 255), 2, 2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, rect


def generate_location(_img, _rect, _obs_width, _obs_heigh, _img_width, _img_heigh):
    # generate new location
    count = 0
    while True:
        count += 1
        new_x = int(random.uniform(_rect[0], _rect[2]))
        new_y = int(random.uniform(_rect[1], _rect[3]))
        if _img[new_y, new_x] == 255:
            new_x = int(new_x - _obs_width/2)
            new_y = int(new_y - _obs_heigh/2)
            if new_x > 0 and new_x < _img_width - _obs_width - 1 and new_y > 0 and new_y < _img_heigh - _obs_heigh - 1:
                return new_x, new_y


        if count == 1000:
            raise "fatual error : image might ganerate no observation area "
            return None

#
# img = cv2.imread("G://pascal_VOC//VOCdevkit//VOC2012//SegmentationClass//2007_004291.png")
# img, rect = connection_area(img, (75, 184))
# x, y = generate_location(img, rect)
#
# cv2.circle(img, (x, y), 3, (0,0,0), 5)
# cv2.imshow("d", img)
# cv2.waitKey(0)


