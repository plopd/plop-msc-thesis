from pathlib import Path

from agents.etd_agent import ETD
from agents.mc_agent import MC
from agents.td_agent import TD
from environments.random_walk_env import RandomWalkEnvironment

PATHS = {"project_path": Path(__file__).parents[2]}

ENVS = {"RandomWalkEnvironment": RandomWalkEnvironment}

AGENTS = {"TD": TD, "ETD": ETD, "MC": MC}
