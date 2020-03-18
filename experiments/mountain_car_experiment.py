import numpy as np

from experiments.experiment import Exp


class MountainCarExp(Exp):
    def __init__(self, agent_info, env_info, experiment_info):
        super().__init__(agent_info, env_info, experiment_info)

    def message(self, message):
        if message == "get approx value":
            current_theta = self.rl_glue.rl_agent_message("get weight vector")
            current_approx_v = np.sum(current_theta[self.state_features], axis=1)
            return current_approx_v
        raise Exception("Unexpected message given.")
