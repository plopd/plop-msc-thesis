from environments.environment import BaseEnvironment


class RandomWalkEnvironment(BaseEnvironment):
    """
    Random Walk Process from Sutton and Barto, 2018
    """

    def __init__(self):
        super(RandomWalkEnvironment, self).__init__()
        self.s_left_term = None
        self.s_right_term = None
        self.r_left = None
        self.r_right = None

    def env_init(self, env_info={}):
        self.N = env_info["N"]
        self.s0 = env_info["s0"]
        self.s_left_term = env_info["s_left_term"]
        self.s_right_term = env_info["s_right_term"]
        self.r_left = env_info["r_left"]
        self.r_right = env_info["r_right"]

    def env_start(self):
        reward = 0.0
        observation = self.s0
        is_terminal = False

        self.reward_obs_term = (reward, observation, is_terminal)

        return self.reward_obs_term[1]

    def env_step(self, action):
        """

        Args:
            action: (int) Either 0 (left) or 1 (right)

        Returns:

        """
        last_state = self.reward_obs_term[1]

        if action == 0:  # transition to left
            current_state = max(self.s_left_term, last_state - 1)
        elif action == 1:  # transition to right
            current_state = min(self.s_right_term, last_state + 1)
        else:
            raise ValueError("Invalid action. Only 0 or 1 are supported.")

        reward = 0.0
        is_terminal = False

        if current_state == self.s_left_term:
            reward = self.r_left
            is_terminal = True
        elif current_state == self.s_right_term:
            reward = self.r_right
            is_terminal = True

        self.reward_obs_term = (reward, current_state, is_terminal)

        return self.reward_obs_term

    def env_cleanup(self):
        pass

    def env_message(self, message):
        pass
