import numpy as np
from agents.agent import BaseAgent
from utils.utils import get_phi


class TDAgent(BaseAgent):
    def __init__(self):
        super(TDAgent, self).__init__()

    def agent_init(self, agent_info):
        self.N = agent_info["N"]
        self.n = agent_info["n"]
        self.phi_repr = agent_info["phi_repr"]
        self.alpha = agent_info["alpha"]
        self.gamma = agent_info["gamma"]
        self.lmbda = agent_info["lmbda"]
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))
        self.phi = get_phi(
            self.N, self.n, seed=agent_info.get("seed"), which=self.phi_repr
        )
        self.theta = np.zeros(self.n)
        self.z = np.zeros_like(self.theta)

        self.s_t = None
        self.a_t = None

    def agent_start(self, observation):

        self.s_t = observation
        self.a_t = self.agent_policy(observation)
        self.z = np.zeros_like(self.theta)

        return self.a_t

    def agent_step(self, reward, observation):
        """

        Args:
            reward:
            observation: (int) in range [1, `self.num_states`]

        Returns:

        """
        current_state_feature = self.phi[observation - 1]
        last_state_feature = self.phi[self.s_t - 1]

        # cf. Eq. 12.5 textbook
        self.z = self.gamma * self.lmbda * self.z + last_state_feature
        # cf. Eq. 12.6 textbook
        td_error = (
            reward
            + self.gamma * np.dot(self.theta.T, current_state_feature)
            - np.dot(self.theta.T, last_state_feature)
        )
        # cf. Eq. 12.7 textbook
        self.theta = self.theta + self.alpha * td_error * self.z

        self.s_t = observation
        self.a_t = self.agent_policy(observation)

        return self.a_t

    def agent_end(self, reward):
        last_state_feature = self.phi[self.s_t - 1]

        # cf. Eq. 12.5 textbook
        self.z = self.gamma * self.lmbda * self.z + last_state_feature
        # cf. Eq. 12.6 textbook
        td_error = reward - np.dot(self.theta.T, last_state_feature)
        # cf. Eq. 12.7 textbook
        self.theta = self.theta + self.alpha * td_error * self.z

        return

    def agent_policy(self, observation):
        """
        Agent policy of taking an action given `observation`.
            Action left (0) or right (1) is taken uniformly at random.
        Args:
            observation:

        Returns:

        """
        return self.rand_generator.choice([0, 1])

    def agent_message(self, message):
        if message == "get state value":
            return np.dot(self.phi, self.theta)
        if message == "get eligibility trace":
            return self.z
        if message == "get feature matrix":
            return self.phi
        if message == "get weight vector":
            return self.theta

    def agent_cleanup(self):
        pass
