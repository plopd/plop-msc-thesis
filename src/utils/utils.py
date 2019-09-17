import os

import numpy as np
from utils.tiles import IHT
from utils.tiles import my_tiles


def path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def calculate_random_feature_matrix(
    num_states, num_features, num_active_features, seed
):

    np.random.seed(seed)
    num_inactive_features = num_features - num_active_features
    representations = np.zeros((num_states, num_features))

    for i_s in range(num_states):
        random_array = np.array([0] * num_inactive_features + [1] * num_active_features)
        np.random.shuffle(random_array)
        representations[i_s, :] = random_array

    return representations


def calculate_phi_for_five_states_with_state_aggregation(num_groups):

    if num_groups == 5:
        group_sizes = [1, 1, 1, 1, 1]
    elif num_groups == 4:
        group_sizes = [2, 1, 1, 1]
    elif num_groups == 3:
        group_sizes = [2, 2, 1]
    elif num_groups == 2:
        group_sizes = [3, 2]
    elif num_groups == 1:
        group_sizes = [5]
    else:
        raise ValueError("Wrong number of groups. Valid are 1, 2, 3, 4 and 5")

    Phi = []
    for i_g, gs in enumerate(group_sizes):
        phi = np.zeros((gs, num_groups))
        phi[:, i_g] = 1.0
        Phi.append(phi)
    Phi = np.concatenate(Phi, axis=0)

    return Phi


def calculate_phi_with_tabular(num_states):
    Phi = np.eye(num_states)
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


def calculate_irmsve(
    true_state_val, learned_state_val, state_distribution, interest, num_states
):
    """

    Args:
        true_state_val:
        learned_state_val:
        state_distribution:
        interest:
        num_states:

    Returns:

    """
    assert len(true_state_val) == len(learned_state_val) == num_states
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

