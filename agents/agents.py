from .etd_agent import ETD
from .mc_agent import MC
from .td_agent import TD


def get_agent(name):
    if name == "td":
        return TD
    elif name == "etd":
        return ETD
    elif name == "mc":
        return MC

    raise Exception("Unexpected agent given")
