import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

import cv2

import simulator_image_getter
import parameters

class FaceSeacherEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):

        self.horizon = simulator_image_getter.Horizon(parameters.HOR_H_MIN, parameters.HOR_H_MAX,
                                                      parameters.HOR_V_MIN, parameters.HOR_V_MAX)
        self.simulator = simulator_image_getter.ImgGetter(parameters.ENV_WIDTH, parameters.ENV_HEIGH,
                                                          parameters.IMG_WIDTH, parameters.IMG_HEIGH,
                                                          self.horizon,
                                                          parameters.BACKGROUND_PATH,
                                                          parameters.PEOPLE_PATH,
                                                          parameters.CLASSFIER_PATH)

        import numpy as np

        self.action_space = spaces.Box(np.array([-parameters.MOVEMENT_MAX,-parameters.MOVEMENT_MAX]),
                                       np.array([parameters.MOVEMENT_MAX, parameters.MOVEMENT_MAX]))

        self.observation_space = spaces.Box(low=0, high=255, shape=(int(parameters.D_HEIGH), int(parameters.D_WIDTH), 1))
        self._seed()

        self.curr_v = self.np_random.uniform(self.horizon.v_horizon_height + parameters.MOVEMENT_MAX*2, self.horizon.v_horizon_low - parameters.MOVEMENT_MAX*2)
        self.curr_h = self.np_random.uniform(self.horizon.h_horizon_height + parameters.MOVEMENT_MAX*2, self.horizon.h_horizon_low - parameters.MOVEMENT_MAX*2)

        self.viewer = None
        self.currImage = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self,u):
        print u
        print "---------------------------"
        self.curr_h = self.curr_h + u[0]
        self.curr_v = self.curr_v + u[1]

        print self.curr_v,self.curr_h
        print self.horizon.v_horizon_low, self.horizon.v_horizon_height
        print self.horizon.h_horizon_low, self.horizon.h_horizon_height
###############################################################################################################################
        if self.curr_h > self.horizon.h_horizon_height or self.curr_v < self.horizon.v_horizon_height\
                or self.curr_h < self.horizon.h_horizon_low or self.curr_v > self.horizon.v_horizon_low:

            print "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
            gray = cv2.cvtColor(self.currImage, cv2.COLOR_RGB2GRAY)
            gray_resized = cv2.resize(gray, (parameters.D_WIDTH, parameters.D_HEIGH))
            return gray_resized, parameters.NEGIVE_REWARD, True, {}

        image, reward = self.simulator.get_image_reward(self.curr_h, self.curr_v, parameters.REWARD_DIS,
                                                        parameters.POSIVE_REWARD, parameters.NEGIVE_REWARD)

        print "sadasdasdasdas",reward

        self.currImage = image

        gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        gray_resized = cv2.resize(gray, (parameters.D_WIDTH,parameters.D_HEIGH))

        return gray_resized, reward, False, {}


    def _reset(self):
        # high = np.array([np.pi, 1])
        # self.state = self.np_random.uniform(low=-high, high=high)
        # self.last_u = None

        self.simulator.generate_new_scence()
        self.curr_v = self.np_random.uniform(self.horizon.v_horizon_low, self.horizon.v_horizon_height)
        self.curr_h = self.np_random.uniform(self.horizon.h_horizon_low, self.horizon.h_horizon_height)
        image, reward = self.simulator.get_image_reward(self.curr_h, self.curr_v, parameters.REWARD_DIS,
                                                        parameters.POSIVE_REWARD, parameters.NEGIVE_REWARD)

        self.currImage = image
        gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        gray_resized = cv2.resize(gray, (parameters.D_WIDTH,parameters.D_HEIGH))

        return gray_resized


###function to show your image
    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if mode == 'rgb_array':
            return self.currImage
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            if  not self.currImage == None:
                self.viewer.imshow(self.currImage)
        return


