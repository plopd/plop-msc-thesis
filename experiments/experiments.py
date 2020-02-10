from experiments.chain_experiment import Chain
from experiments.gridworld_experiment import GridWorldExperiment


def get_experiment(name, agent_info, env_info, exp_info):
    if name == "Chain":
        return Chain(agent_info, env_info, exp_info)
    elif name == "GridWorld":
        return GridWorldExperiment(agent_info, env_info, exp_info)

    raise Exception("Unknown experiment given.")
