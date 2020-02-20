import numpy as np

from agents.ETD import ETD
from agents.TDTileCoding import TDTileCoding


class ETDTileCoding(ETD, TDTileCoding):
    def __init__(self):
        super().__init__()

    def agent_init(self, agent_info):
        super().agent_init(agent_info)

    def agent_end(self, reward):
        return super(TDTileCoding, self).agent_end(reward)

    def learn(self, reward, current_state_feature, last_state_feature):
        target = (
            reward
            if np.isscalar(current_state_feature)
            else reward + self.discount_rate * self.weights[current_state_feature].sum()
        )
        pred = self.weights[last_state_feature].sum()
        delta = target - pred

        self.followon_trace = self.discount_rate * self.followon_trace + self.interest
        self.emphasis = (
            self.trace_decay * self.interest
            + (1 - self.trace_decay) * self.followon_trace
        )
        self.eligibility = self.discount_rate * self.trace_decay * self.eligibility
        self.eligibility[last_state_feature] += self.emphasis

        self.weights += self.step_size * delta * self.eligibility
