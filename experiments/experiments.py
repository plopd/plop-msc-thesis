from experiments.experiment import Exp
from experiments.random_walk_experiment import RandomWalkExp


def get_experiment(name, agent_info, env_info, exp_info):
    if name == "RandomWalk":
        return RandomWalkExp(agent_info, env_info, exp_info)
    elif name == "MountainCar":
        return Exp(agent_info, env_info, exp_info)
    elif name == "PuddleWorld":
        return Exp(agent_info, env_info, exp_info)

    raise Exception("Unknown experiment given.")
