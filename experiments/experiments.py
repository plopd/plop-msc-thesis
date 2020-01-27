from experiments.chain_experiment import ChainExp
from experiments.gridworld_experiment import GridworldExp


def get_experiment(name, agent_info, env_info, exp_info):
    if name == "Chain":
        return ChainExp(agent_info, env_info, exp_info)
    elif name == "Gridworld":
        return GridworldExp(agent_info, env_info, exp_info)

    raise Exception("Unknown experiment given.")
