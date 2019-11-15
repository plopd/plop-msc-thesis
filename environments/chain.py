import numpy as np

from environments.base_environment import BaseEnvironment

LEFT = 0
RIGHT = 1


class Chain(BaseEnvironment):
    """
    Markov Chain from Sutton and Barto, 2018
    """

    def __init__(self):
        super().__init__()
        self.N = None

    def env_init(self, env_info={}):
        self.N = env_info["N"]

    def env_start(self):
        reward = 0
        observation = np.array((self.N // 2,))
        is_terminal = False

        self.reward_obs_term = (reward, observation, is_terminal)

        return self.reward_obs_term[1]

    def env_step(self, action):
        last_state = self.reward_obs_term[1]

        if action == LEFT:
            current_state = np.maximum(-1, last_state - 1)
        elif action == RIGHT:
            current_state = np.minimum(self.N, last_state + 1)
        else:
            raise Exception("Unexpected action given.")

        reward = 0
        is_terminal = False

        if np.array_equal(current_state, np.ones_like(current_state) * -1):
            reward = -1
            is_terminal = True
        elif np.array_equal(current_state, np.ones_like(current_state) * self.N):
            reward = 1
            is_terminal = True

        self.reward_obs_term = (reward, current_state, is_terminal)

        return self.reward_obs_term

    def env_cleanup(self):
        pass

    def env_message(self, message):
        pass
