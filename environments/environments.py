from .chain import Chain
from .deterministic_chain import DeterministicChain


def get_environment(name):
    if name == "chain":
        return Chain
    elif name == "deterministic-chain":
        return DeterministicChain

    raise Exception("Unexpected environment given")
