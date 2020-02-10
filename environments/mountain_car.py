import gym

from environments.base import BaseEnvironment


class MountainCarProblem(BaseEnvironment):
    """Implements MountainCar from https://gym.openai.com/envs/MountainCar-v0/
    """

    def __init__(self):
        super(MountainCarProblem, self).__init__()
        self.env = None

    def env_init(self, env_info={}):
        self.env = gym.make("MountainCar-v0")

    def env_start(self):
        reward = 0
        is_terminal = False
        observation = self.env.reset()
        self.reward_obs_term = (reward, observation, is_terminal)

        return self.reward_obs_term[1]

    def env_step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.reward_obs_term = (reward, observation, done)
        return self.reward_obs_term

    def env_cleanup(self):
        pass
