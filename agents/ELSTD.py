import numpy as np

from agents.ETD import ETD
from agents.LSTD import LSTD


class ELSTD(ETD, LSTD):
    def __init__(self):
        super().__init__()

    def agent_init(self, agent_info):
        super().agent_init(agent_info)

    def learn(self, reward, current_state_feature, last_state_feature):
        self.followon_trace = self.discount_rate * self.followon_trace + self.interest
        self.emphasis = (
            self.trace_decay * self.interest
            + (1 - self.trace_decay) * self.followon_trace
        )
        self.eligibility = (
            self.discount_rate * self.trace_decay * self.eligibility
            + self.emphasis * last_state_feature
        )

        self.A += (
            1
            / self.timesteps
            * (
                np.outer(
                    self.eligibility,
                    last_state_feature - self.discount_rate * current_state_feature,
                )
                - self.A
            )
        )

        self.b += 1 / self.timesteps * (reward * self.eligibility - self.b)

    def agent_cleanup(self):
        super().agent_cleanup()

    def agent_message(self, message):
        response = super().agent_message(message)

        return response
