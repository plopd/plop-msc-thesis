from .MC import MC
from .TDTileCoding import TDTileCoding


class MCTileCoding(MC, TDTileCoding):
    def __init__(self):
        super(MCTileCoding, self).__init__()
        self.trajectory = None
        self.G = None

    def agent_end(self, reward):
        self.trajectory.append((self.s_t, reward))
        for (s_t, r) in self.trajectory[::-1]:
            last_state_feature = self.FR[s_t]
            self.G = self.discount_rate * self.G + r
            pred = self.weights[last_state_feature].sum()
            delta = self.G - pred
            self.weights[last_state_feature] += self.step_size * delta

        return
