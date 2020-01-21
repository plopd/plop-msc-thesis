# import sys
# from pathlib import Path
# import time
# import gym
# import gym_puddle  # noqa f401
# import numpy as np
#
# from utils.utils import path_exists
#
#
# env = gym.make("PuddleWorld-v0", **{"noise": 0})
#
#
# obs = env.reset()
# print(obs)
# done = False
# ACTION = 0
# for i in range(10):
#     observation, reward, _, info = env.step(ACTION)
#     # print("obs, action:", observation, i)
#     print(observation)
#     time.sleep(0.5)
#     env.render("human")
# sys.stdout.flush()
#
# env.close()
