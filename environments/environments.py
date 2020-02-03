from .chain import Chain
from .deterministic_chain import DeterministicChain
from .puddle_world import PuddleWorld


def get_environment(name):
    if name == "chain":
        return Chain
    elif name == "deterministic-chain":
        return DeterministicChain
    elif name == "puddle-world":
        return PuddleWorld

    raise Exception("Unexpected environment given.")
