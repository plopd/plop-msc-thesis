import itertools
import os

import numpy as np
import utils.const as const
import yaml
from rl_glue.rl_glue import RLGlue
from tqdm import tqdm
from utils.tiles import IHT
from utils.tiles import my_tiles


def path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_phi(N, n, num_ones=None, seed=None, which=None):
    if which == "tabular":
        return calculate_phi_with_tabular(N)
    elif which == "random-binary":
        return calculate_phi_with_random_binary_features(N, n, num_ones, seed)
    elif which == "random-non-binary":
        raise NotImplementedError
    elif which == "state-aggregation":
        if N == 5:
            return calculate_phi_for_five_states_with_state_aggregation(n)
        else:
            raise ValueError("State aggregation work only with N=5.")
    else:
        raise ValueError(
            "Unknown feature representation."
            "Only 'tabular', "
            "'random-binary', "
            "'random-non-binary', "
            "'state-aggregation' are valid."
        )


def calculate_phi_with_random_binary_features(N, n, num_ones, seed):

    np.random.seed(seed)
    num_zeros = n - num_ones
    representations = np.zeros((N, n))

    for i_s in range(N):
        random_array = np.array([0] * num_zeros + [1] * num_ones)
        np.random.shuffle(random_array)
        representations[i_s, :] = random_array

    return representations


def calculate_phi_for_five_states_with_state_aggregation(n):

    if n == 5:
        group_sizes = [1, 1, 1, 1, 1]
    elif n == 4:
        group_sizes = [2, 1, 1, 1]
    elif n == 3:
        group_sizes = [2, 2, 1]
    elif n == 2:
        group_sizes = [3, 2]
    elif n == 1:
        group_sizes = [5]
    else:
        raise ValueError("Wrong number of groups. Valid are 1, 2, 3, 4 and 5")

    Phi = []
    for i_g, gs in enumerate(group_sizes):
        phi = np.zeros((gs, n))
        phi[:, i_g] = 1.0
        Phi.append(phi)
    Phi = np.concatenate(Phi, axis=0)

    return Phi


def calculate_phi_with_tabular(N):
    Phi = np.eye(N)
    return Phi


def get_max_size_iht(num_tilings, num_tiles):
    max_size_iht = (num_tiles + 1) * (num_tiles + 1) * num_tilings
    return max_size_iht


def calculate_phi_with_tile_coding(
    num_tilings,
    num_tiles,
    src_left_bound,
    src_right_bound,
    dst_left_bound,
    dst_right_bound,
    num_states,
):
    max_size_iht = get_max_size_iht(num_tilings=num_tilings, num_tiles=num_tiles)
    iht = IHT(max_size_iht)
    feature_matrix = np.zeros((num_states, max_size_iht))
    for idx_state, state in enumerate(range(1, num_states + 1)):
        feature_state = np.zeros(max_size_iht)
        idx_active_tiles = my_tiles(
            iht,
            num_tilings,
            state,
            src_left_bound,
            src_right_bound,
            dst_left_bound,
            dst_right_bound,
        )
        feature_state[idx_active_tiles] = 1
        feature_matrix[idx_state] = feature_state

    return feature_matrix


def calculate_irmsve(true_state_val, learned_state_val, state_distribution, num_states):
    assert len(true_state_val) == len(learned_state_val) == num_states
    interest = np.ones(num_states)
    weighting_factor = np.multiply(state_distribution, interest)
    imsve = np.sum(
        np.multiply(weighting_factor, np.square(true_state_val - learned_state_val))
    )
    imsve_normalized = 1 / np.sum(weighting_factor) * imsve
    irmsve = np.sqrt(imsve_normalized)

    return irmsve


def calculate_auc(ys):
    auc = np.mean(ys)
    return auc


def _zip_with_scalar(l, e):
    return [(e, i) for i in l]


def export_params_from_config_random_walk(cfg):
    with open(f"{const.PATHS['project_path']}/src/configs/{cfg}.yaml", "r") as stream:
        try:
            struct = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    lst = []
    for k, v in struct["agent_info"].items():
        lst.append(_zip_with_scalar(v, k))
    for kk, vv in struct["env_info"].items():
        lst.append(_zip_with_scalar(vv, kk))
    for kkk, vvv in struct["experiment_info"].items():
        lst.append(_zip_with_scalar(vvv, kkk))

    with open(f"{const.PATHS['project_path']}/src/configs/{cfg}.dat", "w") as f:
        for param_conf in itertools.product(*lst):
            line = " ".join(
                [f"{param_key}={param_val}" for (param_key, param_val) in param_conf]
            )
            line = "export " + line
            print(line, file=f)


def _open_yaml_file(filepath):
    with open(filepath, "r") as stream:
        try:
            file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return file


def calculate_true_v(cfg):
    struct = _open_yaml_file(f"{const.PATHS['project_path']}/src/configs/{cfg}.yaml")

    agent_info = struct["agent_info"]
    env_info = struct["env_info"]
    experiment_info = struct["experiment_info"]

    rl_glue = RLGlue(const.ENVS[env_info["env"]], const.AGENTS[agent_info["agent"]])

    rl_glue.rl_init(agent_info, env_info)
    for _ in tqdm(range(1, experiment_info["n_episodes"] + 1)):
        rl_glue.rl_episode(experiment_info["max_timesteps_episode"])

    true_v = rl_glue.rl_agent_message("get state value")

    np.save(
        f"{const.PATHS['project_path']}/data/true_v_"
        f"{struct['agent_info']['N']}_states_random_walk",
        true_v,
    )

    return true_v


def calculate_state_distribution(cfg):
    struct = _open_yaml_file(f"{const.PATHS['project_path']}/src/configs/{cfg}.yaml")

    agent_info = struct["agent_info"]
    env_info = struct["env_info"]
    experiment_info = struct["experiment_info"]

    rl_glue = RLGlue(const.ENVS[env_info["env"]], const.AGENTS[agent_info["agent"]])

    rl_glue.rl_init(agent_info, env_info)

    eta = np.zeros(env_info["N"])
    last_state, _ = rl_glue.rl_start()
    for _ in tqdm(range(1, int(experiment_info["max_timesteps_episode"]) + 1)):
        eta[last_state - 1] += 1
        _, last_state, _, term = rl_glue.rl_step()
        if term:
            last_state, _ = rl_glue.rl_start()

    state_distribution = eta / np.sum(eta)

    np.save(
        f"{const.PATHS['project_path']}/data/state_distribution_"
        f"{struct['agent_info']['N']}_states_random_walk",
        state_distribution,
    )

    return state_distribution
