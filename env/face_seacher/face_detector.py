__author__ = 'Air'
#2016-10-27

import cv2
import math

class FaceDetector():
    def __init__(self, harrcascade_path):
        self.__cascade = cv2.CascadeClassifier(harrcascade_path)
        self.__x = 0
        self.__y = 0
        return

    def find_face(self, _img):
        gray = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        rects = self.__cascade.detectMultiScale(_img)
        assert len(rects) != 0
        max_region = 0
        max_rect = []
        for x1, y1, x2, y2 in rects:
            region = math.fabs((x2 - x1)*(y2 - y1))
            if max_region < region:
                max_rect = [x1, y1, x2, y2]
        self.__x = (max_rect[0] + max_rect[2])/2.0
        self.__y = (max_rect[1] + max_rect[3])/2.0

    def reward(self, _x, _y, _reward_dis, _posive_reward, _negive_reward):
        assert self.__x != 0 and self.__y != 0
        dis = math.sqrt(math.pow(_x - self.__x, 2) + math.pow(_y - self.__y, 2))
        if dis < _reward_dis:
            return _posive_reward
        else:
            return _negive_reward