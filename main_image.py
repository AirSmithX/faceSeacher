import os
import logging
import gym
from gym import envs, scoreboard
from gym.spaces import Discrete, Box
import tempfile
import sys
from environment import Environment
from agent.agent_continous_image import TRPOAgent
import parameters as pms

if not os.path.isdir("./checkpoint"):
    os.makedirs("./checkpoint")
if not os.path.isdir("./log"):
    os.makedirs("./log")
if not os.path.isdir("./img_log"):
    os.makedirs("./img_log")
env = Environment(gym.make(pms.environment_name))
agent = TRPOAgent(env)

if pms.train_flag:
    agent.learn()
else:
    agent.test(pms.checkpoint_file)
# env.monitor.close()
# gym.upload(training_dir,
#            algorithm_id='trpo_ff')
