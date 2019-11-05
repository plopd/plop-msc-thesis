import gym
import gym_puddle  # noqa f401

from environments.base_environment import BaseEnvironment


class PuddleWorld(BaseEnvironment):
    """Implements PuddleWorld from Off-Policy Actor-Critic
    https://arxiv.org/pdf/1205.4839.pdf
    """

    def __init__(self):
        super(PuddleWorld, self).__init__()

    def env_init(self, env_info={}):
        self.env = gym.make("PuddleWorld-v0")

    def env_start(self):
        reward = 0
        is_terminal = False
        observation = self.env.reset()
        self.reward_obs_term = (reward, observation, is_terminal)

        return self.reward_obs_term[1]

    def env_step(self, action):
        observation, reward, done, info = self.env.step(self.env.action_space.sample())
        self.reward_obs_term = (reward, observation, done)

        return self.reward_obs_term

    def env_cleanup(self):
        pass

    def env_message(self, message):
        pass
