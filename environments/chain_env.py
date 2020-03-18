from environments.random_walk_env import RandomWalkEnv


class ChainEnv(RandomWalkEnv):
    def env_start(self):
        return super().env_start()

    def env_step(self, action):
        last_state = self.reward_obs_term[1]

        current_state = last_state + 1

        reward = 0
        is_terminal = False

        if current_state == self.num_states:
            reward = 1
            is_terminal = True

        self.reward_obs_term = (reward, current_state, is_terminal)

        return self.reward_obs_term
