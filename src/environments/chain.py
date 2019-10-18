from environments.base_environment import BaseEnvironment

LEFT = 0
RIGHT = 1


class Chain(BaseEnvironment):
    """
    Markov Chain from Sutton and Barto, 2018
    """

    def __init__(self):
        super(Chain, self).__init__()

    def env_init(self, env_info={}):
        self.N = env_info["N"]

    def env_start(self):
        reward = 0
        observation = self.N // 2 + 1
        is_terminal = False

        self.reward_obs_term = (reward, observation, is_terminal)

        return self.reward_obs_term[1]

    def env_step(self, action):
        last_state = self.reward_obs_term[1]

        if action == LEFT:
            current_state = max(0, last_state - 1)
        elif action == RIGHT:
            current_state = min(self.N + 1, last_state + 1)
        else:
            raise Exception("Unexpected action given")

        reward = 0
        is_terminal = False

        if current_state == 0:
            reward = -1
            is_terminal = True
        elif current_state == self.N + 1:
            reward = 1
            is_terminal = True

        self.reward_obs_term = (reward, current_state, is_terminal)

        return self.reward_obs_term

    def env_cleanup(self):
        pass

    def env_message(self, message):
        pass
