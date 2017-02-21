__author__ = 'Air'
#2016-10-27

import cv2
import os
import random
import face_detector

class Horizon():
    def __init__(self, h_horizon_low, h_horizon_height, v_horizon_low, v_horizon_height):
        self.h_horizon_low = h_horizon_low
        self.h_horizon_height = h_horizon_height
        self.v_horizon_low = v_horizon_height
        self.v_horizon_height = v_horizon_low


class ImgGetter():
    def __init__(self, _env_w, _env_h, _img_w, _img_h, _horizon, _background_path, _people_path, _cascade_path):
        self.__img = []

        self.__env_w = _env_w
        self.__env_h = _env_h
        self.__img_w = _img_w
        self.__img_h = _img_h

        self.__enable_x = _env_w - _img_w
        self.__enable_y = _env_h - _img_h
        self.__fix_x = self.__img_w / 2.0
        self.__fix_y = self.__img_h / 2.0

        self.__horizon = _horizon
        self.__horizon_h_range = _horizon.h_horizon_height - _horizon.h_horizon_low
        self.__horizon_v_range = _horizon.v_horizon_height - _horizon.v_horizon_low

        self.__background_list = []
        backgroud_list = os.listdir(_background_path)
        self.__people_list = []
        people_list = os.listdir(_people_path)

        if len(backgroud_list) > 0:
            for fn in backgroud_list:
                fullfilename = os.path.join(_background_path, fn)
                self.__background_list.append(fullfilename)

        if len(people_list) > 0:
            for fn in people_list:
                fullfilename = os.path.join(_people_path, fn)
                self.__people_list.append(fullfilename)

        self.__background_count = len(self.__background_list)
        self.__people_count = len(self.__people_list)

        self.__face_detector = face_detector.FaceDetector(_cascade_path)
        self.generate_new_scence()


    def get_image_reward(self, curr_h, curr_v, _reward_dis=10, _posive_reward=1, _negive_reward=0):
        curr_x = (curr_h - self.__horizon.h_horizon_low)/self.__horizon_h_range*self.__enable_x + self.__fix_x
        curr_y = (curr_v - self.__horizon.v_horizon_low)/self.__horizon_v_range*self.__enable_y + self.__fix_y
        reward = self.__face_detector.reward(curr_x, curr_y, _reward_dis, _posive_reward, _negive_reward)
        return self.__img[curr_y - self.__fix_y:curr_y + self.__fix_y, curr_x - self.__fix_x:curr_x + self.__fix_x],reward

    def generate_new_scence(self):
        background_number = int(random.uniform(0,self.__background_count))
        people_number = int(random.uniform(0,self.__people_count))

        background = cv2.imread(self.__background_list[background_number])
        people = cv2.imread(self.__people_list[people_number])

        self.__img = background
        for y in range(self.__img.shape[0]):
            for x in range(self.__img.shape[1]):
                if people[y, x, 0] and people[y, x, 1] and people[y, x, 2]:
                    self.__img[y, x] = people[y, x]

        #cv2.imshow()

        self.__face_detector.find_face(self.__img)

#
# hor = Horizon(-30.0, 30.0, -30.0, 40.0)
# imgGetter = ImgGetter(2455.0, 1988.0, 1068.0, 712.0, hor,
#                       "E:\\activereforiencelearing\\train_datas\\backgrounds",
#                       "E:\\activereforiencelearing\\train_datas\\peoples",
#                       "D:/Opencv2411/opencv-2.4.11/data/haarcascades/haarcascade_frontalface_alt.xml")
# img, reward = imgGetter.get_image_reward(-10, 10)
# print reward
# cv2.imshow("show1", img)
#
#
#
# imgGetter.generate_new_scence()
# img, reward = imgGetter.get_image_reward(-20, 0)
# print reward
# cv2.imshow("show2", img)
#
#
# cv2.waitKey(0)

#cv2.imwrite("E:/activereforiencelearing/ttt.jpg",img)