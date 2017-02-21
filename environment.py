"""
`SpaceConversionEnv` acts as a wrapper on
any environment. It allows to convert some action spaces, and observation spaces to others.
"""

import numpy as np
from gym.spaces import Discrete, Box, Tuple
from gym import Env
import cv2
import parameters as pms
import gym
from gym.monitoring import monitor

def convert_gym_space(space):
    if isinstance(space, gym.spaces.Box):
        return Box(low=space.low, high=space.high)
    elif isinstance(space, gym.spaces.Discrete):
        return Discrete(n=space.n)
    else:
        raise NotImplementedError

class CappedCubicVideoSchedule(object):
    def __call__(self, count):
        return monitor.capped_cubic_video_schedule(count)

class NoVideoSchedule(object):
    def __call__(self , count):
        return False

class Environment(Env):

    def __init__(self, env, type="origin"):
        self.env = env
        self.type = type
        self.video_schedule = None
        if not pms.record_movie:
            self.video_schedule = NoVideoSchedule()
        else:
            if self.video_schedule is not None:
                self.video_schedule = CappedCubicVideoSchedule()
            self.env.monitor.start("log/trpo" ,self.video_schedule, force=True)
            self.monitoring = True

    def step(self, action, **kwargs):
        self._observation, reward, done, info = self.env.step(action)
        # self._observation[0] = np.clip(self._observation[0], self.env.observation_space.low, self.env.observation_space.high)
        return self.observation, reward, done, info

    def reset(self, **kwargs):
        self._observation = self.env.reset()
        return self.observation

    def render(self, mode="human", close=False):
        return self.env.render(mode)

    @property
    def observation(self):
        if self.type == "origin":
            return self._observation
        elif self.type == "gray_image":
            return cv2.resize(cv2.cvtColor(self._observation, cv2.COLOR_RGB2GRAY)/255., pms.dims)

    @property
    def action_space(self):
        return convert_gym_space(self.env.action_space)


    @property
    def observation_space(self):
        if self.type == "origin":
            return convert_gym_space(self.env.observation_space)
        else:
            return pms.dims

    # @property
    # def obs_dims(self):
    #     if self.type == "origin":
    #         return self.env.observation_space.shape
    #     else:
    #         return pms.dims