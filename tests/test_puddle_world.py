import sys  # noqa f401
import time  # noqa f401
from pathlib import Path  # noqa f401

import gym  # noqa f401
import gym_puddle  # noqa f401
import numpy as np  # noqa f401
from tqdm import tqdm  # noqa f401

from agents.policies import get_action_from_policy  # noqa f401
from utils.utils import path_exists  # noqa f401

# env = gym.make("PuddleWorld-v0")

# # Determine action mapping
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

# # Simulate on-policy distribution
# seed = 0
# rand_generator = np.random.RandomState(seed)
# env.seed(seed)
# env.reset()
# done = False
# states = []
# timesteps = 10**7
# for t in tqdm(range(timesteps)):
#     action = get_action_from_policy("semi-random-puddle", rand_generator)
#     observation, reward, done, info = env.step(action)
#     states.append(observation)
#     # env.render("human")
#     if done:
#         # print("Done. Reached.", observation)
#         env.reset()
# env.close()
# np.save("/Users/saipiens/scratch/PuddleWorld/states", states, allow_pickle=True)

# # Sample 500 states from the last half of timesteps
# states = np.load("/Users/saipiens/scratch/PuddleWorld/states.npy", allow_pickle=True)
# idxs = np.random.choice(np.arange(len(states)//2, len(states)), size=(500,),
#                         replace=False)
# S = states[idxs, :]
# states_puddle = np.save("/Users/saipiens/scratch/PuddleWorld/states_puddle", S,
#                         allow_pickle=True)


# # Plot states S
# import matplotlib.pyplot as plt
#
# plt.scatter(S[:, 0], S[:, 1], alpha=0.25, label="S")
# plt.scatter([0.2], [0.4], label="start", marker="o", s=100)
# plt.scatter([1.0], [1.0], label="goal", marker="^", s=100)
# plt.xticks(ticks=[.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.],
#            labels=[.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])
# plt.yticks(ticks=[.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.],
#            labels=[.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])
# plt.xlim((0., 1.08))
# plt.ylim((0., 1.08))
# plt.legend()
# plt.show()

# # Get true values by averaging returns

# gamma = 1.
# n_episodes = 1000
# states_puddle = np.load("/Users/saipiens/scratch/PuddleWorld/states_puddle.npy",
#                         allow_pickle=True)
#
# true_values = np.zeros(len(states_puddle))
#
# for i in tqdm(range(len(states_puddle))):
#     state = states_puddle[i]
#     env = gym.make("PuddleWorld-v0", **{"start": state})
#     Gs = []
#     for n_e in range(n_episodes):
#         rand_generator = np.random.RandomState(n_e)
#         rewards = []
#         done = False
#         obs = env.reset()
#         while not done:
#             action = get_action_from_policy("semi-random-puddle", rand_generator)
#             observation, reward, done, info = env.step(action)
#             rewards.append(reward)
#             # env.render("human")
#             if done:
#                 G = np.sum([gamma ** i * r for i, r in enumerate(rewards)])
#                 Gs.append(G)
#     true_value = np.mean(Gs)
#     print(f"s0: {state}, true value: {true_value}")
#     true_values[i] = true_value
#
# np.save("/Users/saipiens/scratch/PuddleWorld/true_v_puddle", true_values, allow_pickle=True)
