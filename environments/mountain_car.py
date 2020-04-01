import gym

from environments.base import BaseEnvironment


class MountainCarEnv(BaseEnvironment):
    """Wrapper for RL-glue for MountainCar from https://gym.openai.com/envs/MountainCar-v0/
    """

    def __init__(self):
        super(MountainCarEnv, self).__init__()
        self.env = None

    def env_init(self, env_info={}):
        self.log_episodes = env_info.get("log_episodes")
        self.env = gym.make("MountainCar-v0")
        self.env.seed(env_info.get("seed"))

    def env_start(self):
        if self.log_episodes:
            self.experience_episode = []
        reward = 0
        is_terminal = False
        observation = self.env.reset()
        self.reward_obs_term = (reward, observation, is_terminal)

        return self.reward_obs_term[1]

    def env_step(self, action):
        last_state = self.reward_obs_term[1]
        if self.log_episodes:
            self.experience_episode.append(last_state)
        observation, reward, done, info = self.env.step(action)
        self.reward_obs_term = (reward, observation, done)
        return self.reward_obs_term

    def env_message(self, message):
        if message == "get episode" and self.log_episodes:
            return self.experience_episode
