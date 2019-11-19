from .elstd_agent import ELSTD
from .etd_agent import ETD
from .lstd_agent import LSTD
from .mc_agent import MC
from .td_agent import TD


def get_agent(name):
    if name == "td":
        return TD
    elif name == "etd":
        return ETD
    elif name == "mc":
        return MC
    elif name == "lstd":
        return LSTD
    elif name == "elstd":
        return ELSTD

    raise Exception("Unexpected agent given")
