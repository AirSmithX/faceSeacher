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
import parameters as pm

import random
import time


#out_type
# 0 stands for normal
# 1 stands for out_of_range
# 2 stands for success

# load a img and find the face
# rember the face_location
# action would move the location
class FaceSeacherEnv_v2(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
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
        self.get_all_file_names(pm.FILE_DIR)

        self.generate_new_scence()

        # #action space first array is the min_action space loc, second array is the hight action space
        # self.action_space = spaces.Box(np.array([-pm.MOVEMENT_MAX,-pm.MOVEMENT_MAX]),
        #                                np.array([pm.MOVEMENT_MAX, pm.MOVEMENT_MAX]))
        #we add a action that can resize the observation window
        ###TODO it seems noscence the network's output actually puts no bounded action not bounded by this actionspace
        self.action_space = spaces.Box(np.array([0, 1]),
                                       np.array([1, 2]))

        #return a 300X300 BGR image
        self.observation_space = spaces.Box(low=0, high=255, shape=(int(pm.OBS_BOX_HEIGH), int(pm.OBS_BOX_WIDTH), 3))

        random.seed(time.time())


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self,u):

        u = u * pm.MOVEMENT_MAX

        self.step_count = self.step_count+1

        # print (int(self.curr_x + pm.OBS_BOX_WIDTH / 2), int(self.curr_y + pm.OBS_BOX_HEIGH / 2)),(int(self.curr_x + pm.OBS_BOX_WIDTH / 2 + u[0]), int(self.curr_y + pm.OBS_BOX_HEIGH / 2 + u[1]))

        self.curr_x_old = self.curr_x
        self.curr_y_old = self.curr_y

        self.curr_x = int(self.curr_x + u[0])
        self.curr_y = int(self.curr_y + u[1])

        cv2.line(self.draw_img, (self.curr_x_old + pm.OBS_BOX_WIDTH /2,self.curr_y_old+ pm.OBS_BOX_WIDTH /2),(self.curr_x+ pm.OBS_BOX_WIDTH /2,self.curr_y+ pm.OBS_BOX_WIDTH /2),(0,0,255),thickness=30)

        # print "curr:" , self.curr_x, self.curr_y

        # weather the curr_location is over the boundry
        if self.curr_x < 0 or self.curr_x > (self.__env_img.shape[1] - pm.OBS_BOX_WIDTH- 1 ) \
            or self.curr_y < 0 or self.curr_y > (self.__env_img.shape[0] - pm.OBS_BOX_HEIGH -1):
            #over boungry
            print "out of boundry"
            self.out_type = 1
            self.reward_sum = self.reward_sum - 100
            self.step_reward.append(-100)
            return  np.array([self.obs_img,[[self.curr_x/float(pm.ENV_WIDTH),self.curr_y/float(pm.ENV_HEIGH)]]]), -100, True, {}

        # cancutate  a new conitnus reward

        rewards = self.canculate_rewards()
        self.obs_img = self.__env_img[self.curr_y: self.curr_y + pm.OBS_BOX_HEIGH, self.curr_x: self.curr_x + pm.OBS_BOX_WIDTH]

        if -rewards < -9.9:
            self.out_type = 2
            self.reward_sum = self.reward_sum + 100
            self.step_reward.append(100)
            return np.array([self.obs_img,[[self.curr_x/float(pm.ENV_WIDTH),self.curr_y/float(pm.ENV_HEIGH)]]]) , rewards + 100, True, {}
        self.reward_sum = self.reward_sum + rewards
        self.out_type = 0
        self.step_reward.append(rewards)
        return  np.array([self.obs_img,[[self.curr_x/float(pm.ENV_WIDTH),self.curr_y/float(pm.ENV_HEIGH)]]]) , rewards, False, {}

    def _reset(self):
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
        return np.array([self.obs_img,[[self.curr_x/float(pm.ENV_WIDTH),self.curr_y/float(pm.ENV_HEIGH)]]])

    ###function to show your image
    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if mode == 'rgb_array':
            return self.obs_img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            if  not self.obs_img == None:
                self.viewer.imshow(self.obs_img)
        return

    def get_all_file_names(self, _path):
        file_list = os.listdir(_path)

        if file_list:
            for fn in file_list:
                full_file_name = os.path.join(_path, fn)
                self.__full_file_lists.append(full_file_name)

    def generate_new_scence(self):
        file_name = self.__full_file_lists[int(random.uniform(0,len(self.__full_file_lists)))]
        print "generate file from: :",file_name
        self.__env_img = cv2.imread(file_name)
        #inital the box
        ################################
        self.curr_x = int(random.uniform(0, int(self.__env_img.shape[1] - pm.OBS_BOX_WIDTH - 1)))
        self.curr_y = int(random.uniform(0, int(self.__env_img.shape[0] - pm.OBS_BOX_HEIGH - 1)))

        img = cv2.cvtColor(self.__env_img, cv2.COLOR_BGR2RGB)
        self.__dets = self.__detector(img, 1)
        self.obs_img = self.__env_img[self.curr_y : self.curr_y + pm.OBS_BOX_HEIGH, self.curr_x : self.curr_x + pm.OBS_BOX_WIDTH]
        self.draw_img = self.__env_img.copy()
        cv2.circle(self.draw_img,(self.curr_x+ pm.OBS_BOX_WIDTH /2,self.curr_y+ pm.OBS_BOX_WIDTH /2),60,(0,255,0),20)


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
        dis_reward_max = 0
        area_reward_max = 0

        for i, det in enumerate(self.__dets):
            # overlap_area = self.canculate_rec_area((det.left(),det.top()),(det.right(),det.bottom()),
            #                                        (self.curr_x,self.curr_y),(self.curr_x+pm.OBS_BOX_WIDTH,self.curr_y+pm.OBS_BOX_HEIGH))


            # print "det: ", det.right(),det.left(),det.bottom(),det.top()
            # print "box: ", self.curr_x,self.curr_y, self.curr_x+pm.OBS_BOX_WIDTH, self.curr_y + pm.OBS_BOX_HEIGH,'\n'
            #
            # # w_max = max([det.right() - self.curr_x, self.curr_x + pm.OBS_BOX_WIDTH - det.left()])
            # # h_max = max([det.bottom() - self.curr_y, self.curr_y + pm.OBS_BOX_HEIGH - det.top()])
            #
            #
            #
            # print w_max,h_max,"\n"
            #
            # if w_max > 0 and h_max > 0:
            #     overlap_area = w_max * h_max
            #     det_area = (det.right() - det.left()) * (det.bottom() - det.top())
            #     area_reward = float(overlap_area) / float(det_area)
            #     # rewards += area_reward
            #
            #     print overlap_area, det_area, area_reward
            #
            #     if area_reward_max < area_reward:
            #         area_reward_max = area_reward
            #

            # if overlap_area > 0:
            #     det_area = (det.right() - det.left()) * (det.bottom() - det.top())
            #     area_reward = float(overlap_area) / float(det_area)
            #     if area_reward_max < area_reward:
            #         area_reward_max = area_reward


            center_x = (det.left() + det.right())/2
            center_y = (det.bottom() + det.top())/2
            center_obs_x = self.curr_x + pm.OBS_BOX_WIDTH /2
            center_obs_y = self.curr_y + pm.OBS_BOX_HEIGH /2

            center_dis = math.sqrt(math.pow(center_x - center_obs_x, 2) + math.pow(center_y - center_obs_y, 2))
            dis_reward = math.exp(pm.DECLAY * center_dis)*10
            # rewards += area_reward
            if dis_reward_max < dis_reward:
                dis_reward_max = dis_reward

        return dis_reward_max+area_reward_max

    def save_draw_image(self):
        self.draw_count += 1
        # if self.draw_count > 1000:
        #     self.draw_count = 0
        if self.draw_count % 100 == 0:
            cv2.circle(self.draw_img, (self.curr_x+ pm.OBS_BOX_WIDTH /2, self.curr_y+ pm.OBS_BOX_WIDTH /2), 60, (255, 0, 0), 20)
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
            cv2.putText(self.draw_img,text_string,(80,80),cv2.FONT_HERSHEY_COMPLEX,3,(0,0,255),5)
            cv2.imwrite("./img_log/"+str(self.draw_count)+".jpg", self.draw_img)













