import gym
import gym_puddle  # noqa f401

from environments.base import BaseEnvironment


class PuddleWorld(BaseEnvironment):
    """Implements PuddleWorld from Off-Policy Actor-Critic
    https://arxiv.org/pdf/1205.4839.pdf

    ACTIONS:
        0 - west
        1 - east
        2 - south
        3 - north
        4 - don't move
    """

    def __init__(self):
        super(PuddleWorld, self).__init__()
        self.env = None
        self.len_episode = None

    def env_init(self, env_info={}):
        self.env = gym.make("PuddleWorld-v0")
        self.env.seed(env_info.get("seed"))
        self.len_episode = 0

    def env_start(self):
        reward = 0
        is_terminal = False
        observation = self.env.reset()
        self.len_episode = 0
        self.reward_obs_term = (reward, observation, is_terminal)

        return self.reward_obs_term[1]

    def env_step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.reward_obs_term = (reward, observation, done)
        self.len_episode += 1
        return self.reward_obs_term

    def env_cleanup(self):
        pass

    def env_message(self, message):
        if message == "get length episode":
            return self.len_episode
        raise Exception("Unexpected environment message given.")
