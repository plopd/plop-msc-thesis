import numpy as np

from agents.td_agent import TD


class LSTD(TD):
    def __init__(self):
        super().__init__()
        self.A = None
        self.b = None
        self.steps = None

    def agent_init(self, agent_info):
        super().agent_init(agent_info)

        self.steps = 0
        num_features = self.feature.shape[1]
        self.A = np.zeros((num_features, num_features))
        self.b = np.zeros((num_features, 1))

        return self.a_t

    def agent_start(self, observation):
        self.a_t = super().agent_start(observation)
        self.steps = 1
        return self.a_t

    def agent_step(self, reward, observation):
        self.a_t = super().agent_step(reward, observation)
        self.steps += 1

        return self.a_t

    def _learn(self, reward, current_state_feature, last_state_feature):
        self.z = self.gamma * self.lmbda * self.z + last_state_feature

        self.A += (
            1
            / self.steps
            * (
                np.dot(
                    np.expand_dims(self.z, axis=1),
                    np.expand_dims(
                        last_state_feature - self.gamma * current_state_feature, axis=1
                    ).T,
                )
                - self.A
            )
        )

        self.b += 1 / self.steps * (reward * self.z - self.b)
