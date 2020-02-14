import numpy as np

from agents.TD import TD


class LSTD(TD):
    def __init__(self):
        super().__init__()
        self.A = None
        self.b = None
        self.timesteps = None

    def agent_init(self, agent_info):
        super().agent_init(agent_info)
        self.timesteps = 0
        self.A = np.zeros((self.FR.num_features, self.FR.num_features))
        self.b = np.zeros((self.FR.num_features,))

    def agent_start(self, observation):
        self.a_t = super().agent_start(observation)
        return self.a_t

    def agent_step(self, reward, observation):
        self.timesteps += 1
        self.a_t = super().agent_step(reward, observation)

        return self.a_t

    def agent_cleanup(self):
        super().agent_cleanup()

    def learn(self, reward, current_state_feature, last_state_feature):
        self.eligibility = (
            self.discount_rate * self.trace_decay * self.eligibility
            + last_state_feature
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

    def agent_message(self, message):
        if message == "get A":
            return self.A
        elif message == "get b":
            return self.b
        elif message == "get weight vector":
            try:
                self.weights = np.dot(np.linalg.inv(self.A), self.b)
            except np.linalg.LinAlgError:
                return self.weights
        elif message == "get steps":
            return self.timesteps
        response = super().agent_message(message)

        return response
