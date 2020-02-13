from agents.ETD import ETD
from agents.TDTileCoding import TDTileCoding


class ETDTileCoding(ETD, TDTileCoding):
    def agent_end(self, reward):
        return super(TDTileCoding, self).agent_end(reward)

    def learn(self, reward, current_state_feature, last_state_feature):
        target = (
            reward
            if not current_state_feature
            else reward + self.gamma * self.weights[current_state_feature].sum()
        )
        pred = self.weights[last_state_feature].sum()
        delta = target - pred

        self.F = self.gamma * self.F + self.i
        self.M = self.lmbda * self.i + (1 - self.lmbda) * self.F
        self.eligibility = self.gamma * self.lmbda * self.eligibility
        self.eligibility[last_state_feature] += self.M

        self.weights += (self.alpha / self.FR.tilings) * delta * self.eligibility
