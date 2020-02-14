import numpy as np

from agents.TD import TD


class ETD(TD):
    def __init__(self):
        super().__init__()
        self.interest = None
        self.followon_trace = None
        self.emphasis = None

    def agent_message(self, message):
        if message == "get followon trace":
            return self.followon_trace
        elif message == "get emphasis trace":
            return self.emphasis
        response = super().agent_message(message)
        return response

    def learn(self, reward, current_state_feature, last_state_feature):
        target = reward + self.discount_rate * np.dot(
            self.weights.T, current_state_feature
        )
        pred = np.dot(self.weights.T, last_state_feature)
        delta = target - pred

        self.followon_trace = self.discount_rate * self.followon_trace + self.interest
        self.emphasis = (
            self.trace_decay * self.interest
            + (1 - self.trace_decay) * self.followon_trace
        )
        self.eligibility = (
            self.discount_rate * self.trace_decay * self.eligibility
            + self.emphasis * last_state_feature
        )
        self.weights += self.step_size * delta * self.eligibility

    def agent_cleanup(self):
        super().agent_cleanup()
        self.interest = 1.0
        self.followon_trace = 0.0
        self.emphasis = 0.0
