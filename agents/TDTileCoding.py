import numpy as np

from agents.TD import TD


class TDTileCoding(TD):
    def __init__(self):
        super().__init__()

    def agent_init(self, agent_info):
        super().agent_init(agent_info)

        self.step_size /= self.FR.tilings

    def learn(self, reward, current_state_feature, last_state_feature):
        target = (
            reward
            if np.isscalar(current_state_feature)
            else reward + self.discount_rate * self.weights[current_state_feature].sum()
        )
        pred = self.weights[last_state_feature].sum()
        self.eligibility = self.discount_rate * self.trace_decay * self.eligibility
        self.eligibility[last_state_feature] += 1
        self.weights += self.step_size * (target - pred) * self.eligibility
