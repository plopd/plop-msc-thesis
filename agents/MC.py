import numpy as np

from agents.TD import TD


class MC(TD):
    def __init__(self):
        super(MC, self).__init__()
        self.trajectory = None
        self.G = None

    def agent_init(self, agent_info):
        super(MC, self).agent_init(agent_info)
        self.trajectory = []
        self.G = 0

    def agent_start(self, observation):
        self.a_t = super(MC, self).agent_start(observation)
        self.G = 0
        self.trajectory = []

        return self.a_t

    def agent_step(self, reward, observation):
        self.trajectory.append((self.s_t, reward))

        self.s_t = observation
        self.a_t = self.agent_policy(observation)

        return self.a_t

    def agent_policy(self, observation):
        return super(MC, self).agent_policy(observation)

    def agent_end(self, reward):
        self.trajectory.append((self.s_t, reward))
        for (s_t, r) in self.trajectory[::-1]:
            self.G = self.gamma * self.G + r
            delta = self.G - np.dot(self.theta.T, self.feature_type[s_t])
            self.theta = self.theta + self.alpha * delta * self.feature_type[s_t]

        return

    def agent_message(self, message):
        if message == "get episode":
            return self.trajectory
        elif message == "get return":
            return self.G
        response = super(MC, self).agent_message(message)
        return response

    def agent_cleanup(self):
        pass
