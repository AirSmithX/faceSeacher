import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

import os

import dlib
import cv2
import math

import simulator_image_getter
import paramaters_v3 as pm

import random
import time


import face_seacher_utils as ut
import xml_ananysizer


#out_type
# 0 stands for normal
# 1 stands for out_of_range
# 2 stands for success

# load a img and find the face
# rember the face_location
# action would move the location




class FaceSeacherEnv_v3(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }
    def set_scope(self,scope):
        self.scope = scope

    def __init__(self):
        self.scope = "scope"
        self.reward_old = None
        self.xml_phrazer = xml_ananysizer.xmlPhrazer()

        #open the data_log file , and count how many records have record
        self.first_flag = True

        self.data_log = open("./log/data_log.txt")
        count = 0
        line = self.data_log.readline()
        while line:
            line = self.data_log.readline()
            count = count + 1

        self.draw_count = count
        self.data_log.close()


        self.out_type = 0
        self.reward_sum = 0
        self.step_count = 0
        self.step_reward = []


        # self.draw_count = self.inital_number
        self.color_count = 0
        self.viewer = None
        self.obs_img = None
        self.__detector = dlib.get_frontal_face_detector()
        self.__full_file_lists = []
        self.get_all_file_names(pm.IMG_FILES_PATH)

        self.generate_new_scence()

        # #action space first array is the min_action space loc, second array is the hight action space
        # self.action_space = spaces.Box(np.array([-pm.MOVEMENT_MAX,-pm.MOVEMENT_MAX]),
        #                                np.array([pm.MOVEMENT_MAX, pm.MOVEMENT_MAX]))
        #we add a action that can resize the observation window
        ###TODO it seems noscence the network's output actually puts no bounded action not bounded by this actionspace
        self.action_space = spaces.Box(np.array([0, 1]),
                                       np.array([1, 2]))

        #return a 300X300 BGR image
        self.observation_space = spaces.Box(low=0, high=255, shape=(100, 100, 3))

        random.seed(time.time())


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def decrease_draw_count(self):
        self.draw_count -= 1

    def _step(self,u):
        u = np.clip(u,-2,2)

        # u = u * (pm.MOVEMENT_SCALE * self.obs_width)


        self.step_count = self.step_count+1

        self.curr_x_old = self.curr_x
        self.curr_y_old = self.curr_y

        self.curr_x = int(self.curr_x + u[0])
        self.curr_y = int(self.curr_y + u[1])


        # print "curr:" , self.curr_x, self.curr_y

        # weather the curr_location is over the boundry

        self.curr_x = np.clip(self.curr_x, 0, (self.env_width - self.obs_width - 1 ))
        self.curr_y = np.clip(self.curr_y, 0, (self.env_heigh - self.obs_heigh - 1 ))

        cv2.line(self.draw_img, (self.curr_x_old + self.obs_width /2,self.curr_y_old+ self.obs_heigh /2),(self.curr_x+ self.obs_width /2,self.curr_y+ self.obs_heigh /2),(0,0,255),thickness=6)


        # if self.curr_x < 0 or self.curr_x > (self.env_width - self.obs_width - 1 ) \
        #     or self.curr_y < 0 or self.curr_y > (self.env_heigh - self.obs_heigh -1) \
        #     or (self.curr_x + self.obs_width/2) < self.__rect[0] or (self.curr_x + self.obs_width/2) > self.__rect[2] \
        #     or (self.curr_y + self.obs_heigh/2) < self.__rect[1] or (self.curr_y + self.obs_heigh/2) > self.__rect[3]:
        #     #over boungry
        #     self.out_type = 1
        #     self.reward_sum = self.reward_sum -(-pm.MAX_REWARD+ pm.BAIS_LEVEL)
        #     self.step_reward.append(-(-pm.MAX_REWARD+ pm.BAIS_LEVEL))
            # print "outtype 1  ",(pm.MAX_REWARD+ pm.BAIS_LEVEL)

            # return  self.obs_img, -(-pm.MAX_REWARD+ pm.BAIS_LEVEL), True, {}

        # cancutate  a new conitnus reward

        rewards = self.canculate_rewards()
        obs_img = self.__env_img[self.curr_y: self.curr_y + self.obs_heigh, self.curr_x: self.curr_x + self.obs_width]
        self.obs_img = cv2.resize(cv2.cvtColor(obs_img, cv2.COLOR_RGB2GRAY),(100,100))


        actual_reward = 0
        if  not self.reward_old == None:
            actual_reward = rewards - self.reward_old
            self.reward_old = rewards
        else:
            actual_reward = 0.01
            self.reward_old = rewards


        if actual_reward > pm.MAX_REWARD:
            actual_reward = pm.MAX_REWARD
        if actual_reward < -pm.MAX_REWARD:
            actual_reward = -pm.MAX_REWARD


        if rewards > 90:
            self.out_type = 2
            self.reward_sum = self.reward_sum  + (pm.MAX_REWARD+ pm.BAIS_LEVEL +pm.SUCCESS_BAIS)
            self.step_reward.append( (pm.MAX_REWARD+ pm.BAIS_LEVEL +pm.SUCCESS_BAIS))
            # print "outtype 2  ",  (- pm.MAX_REWARD+ pm.BAIS_LEVEL)
            return self.obs_img,  (pm.MAX_REWARD+ pm.BAIS_LEVEL +pm.SUCCESS_BAIS), True, {}

        self.reward_sum = self.reward_sum +(actual_reward+pm.BAIS_LEVEL)
        self.out_type = 0
        self.step_reward.append(actual_reward+pm.BAIS_LEVEL)
        # print "outtype 0  ", (- actual_reward+pm.BAIS_LEVEL)
        return self.obs_img,actual_reward+pm.BAIS_LEVEL, False, {}

    def _reset(self):
        self.reward_old = None
        if self.first_flag:
            self.first_flag = False
        else:
            self.data_log = open("./log/data_log.txt", 'a')
            self.data_log.write(str(self.out_type)+ ";" + str(self.step_count)+";"+str(self.reward_sum)+'\n')
            self.data_log.close()
            self.save_draw_image()
        self.out_type = 0
        self.reward_sum = 0
        self.step_count = 0
        self.step_reward = []

        self.generate_new_scence()
        # cv2.imshow("aaa", self.obs_img)
        # cv2.waitKey(0)
        return self.obs_img

    ###function to show your image
    def _render(self, mode='human', close=False):
        # if close:
        #     if self.viewer is not None:
        #         self.viewer.close()
        #         self.viewer = None
        #     return
        #
        # if mode == 'rgb_array':
        #     return self.obs_img
        # elif mode == 'human':
        #     from gym.envs.classic_control import rendering
        #     if self.viewer is None:
        #         self.viewer = rendering.SimpleImageViewer()
        #     if  not self.obs_img == None:
        #         self.viewer.imshow(self.obs_img)
        return

    def get_all_file_names(self, _path):
        file_list = os.listdir(_path)

        if file_list:
            for fn in file_list:
                full_file_name = os.path.join(_path, fn)
                self.__full_file_lists.append(full_file_name)

    def generate_new_scence(self):
        while True:
            file_name = self.__full_file_lists[int(random.uniform(0, len(self.__full_file_lists)))]
            # file_name = "/home/air/face_seacher_train_dir/washed_imgs/2009_002152.jpg"
            print "generate file from: :", file_name
            self.__env_img = cv2.imread(file_name)
            self.env_heigh, self.env_width = self.__env_img.shape[:2]

            ##canculate the connection area
            xml_name = pm.XML_FILES_PATH + '/' + file_name.split('/')[-1].split('.')[0] + ".xml"
            self.__rect = self.xml_phrazer.phraze_xml(xml_name)
            if self.__rect:
                # find a face
                img = cv2.cvtColor(self.__env_img, cv2.COLOR_BGR2RGB)
                dets = self.__detector(img, 1)
                if dets:
                    self.__det = dets[0]
                    self.obs_width = int((self.__det.right() - self.__det.left()) * pm.OBS_SCALE)
                    self.obs_heigh = int((self.__det.bottom() - self.__det.top()) * pm.OBS_SCALE)
                    if self.obs_width < self.obs_heigh:
                        self.obs_heigh = self.obs_width
                    else:
                        self.obs_width = self.obs_heigh

                    # self.__seg_img, self.__rect = ut.connection_area(self.__seg_img, (center_x, center_y))
                    # self.curr_x, self.curr_y = ut.generate_location(self.__seg_img, self.__rect, self.obs_width, self.obs_heigh, self.env_width, self.env_heigh)
                    # rect = self.xml_phrazer()
                    self.curr_x = int(random.uniform(self.__rect[0], self.__rect[2] - self.obs_width))
                    self.curr_y = int(random.uniform(self.__rect[1], self.__rect[3] - self.obs_heigh))

                    self.obs_img = self.__env_img[self.curr_y: self.curr_y + self.obs_heigh,
                                   self.curr_x: self.curr_x + self.obs_width]

                    self.draw_img = self.__env_img.copy()
                    cv2.circle(self.draw_img, (self.curr_x + self.obs_heigh / 2, self.curr_y + self.obs_width / 2), 20,
                               (0, 255, 0), 12)
                    if self.obs_img.shape[0] > 0 and self.obs_img.shape[1] > 0:
                        self.obs_img = cv2.resize(cv2.cvtColor(self.obs_img, cv2.COLOR_RGB2GRAY), (100, 100))
                        self.declay = -1.0/(self.obs_width * 4.0)
                        break
                    else:
                        pass
                else:
                    pass
            else:
                pass


    def canculate_rec_area(self,rec_1_p1,rec_1_p2,rec_2_p1,rec_2_p2):
        line_1 = self.canculate_one_line(rec_1_p1[0],rec_1_p2[0],rec_2_p1[0],rec_2_p2[0])
        line_2 = self.canculate_one_line(rec_1_p1[1],rec_1_p2[1],rec_2_p1[1],rec_2_p2[1])
        return line_1*line_2

    def canculate_one_line(self,left_1,right_1,left_2,right_2):
        max_point = max([right_1,right_2])
        min_point = min([left_1,left_2])

        step_foot = []
        for step in range(min_point,max_point+1):
            if step > left_1 and step < right_1 and step > left_2 and step < right_2:
                step_foot.append(step)
        if step_foot:
            return step_foot[-1] - step_foot[0]
        else:
            return 0


    def canculate_rewards(self):
        center_x = (self.__det.left() + self.__det.right())/2
        center_y = (self.__det.bottom() + self.__det.top())/2
        center_obs_x = self.curr_x + self.obs_width /2
        center_obs_y = self.curr_y + self.obs_heigh /2

        center_dis = math.sqrt(math.pow(center_x - center_obs_x, 2) + math.pow(center_y - center_obs_y, 2))
        dis_reward = math.exp(self.declay * center_dis)*100
        # rewards += area_reward
        return  dis_reward

    def save_draw_image(self):
        self.draw_count += 1

        if self.draw_count % pm.DRAW_NUMBER == 0:
            cv2.circle(self.draw_img, (self.curr_x+ self.obs_heigh /2, self.curr_y+ self.obs_width /2), 20, (255, 0, 0), 6)
            text_string = ""
            if self.out_type == 0:
                text_string+="NML"
            if self.out_type == 1:
                text_string+="FIL"
            if self.out_type == 2:
                text_string+="SUC"
            text_string+="   "
            text_string+=str(self.step_count)
            text_string+="   "
            text_string+=str(self.reward_sum)
            cv2.putText(self.draw_img,text_string,(0,20),cv2.FONT_ITALIC,0.7,(0,0,255),2)
            cv2.imwrite("./img_log/"+self.scope+"_"+str(self.draw_count)+".jpg", self.draw_img)