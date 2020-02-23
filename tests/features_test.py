import numpy as np
import pytest

from representations.representations import get_representation
from utils.utils import per_feature_step_size_fourier_KOT


@pytest.mark.parametrize("num_states", [5, 19])
def test_tabular_features(num_states):
    states = np.arange(num_states).reshape(-1, 1)
    TF = get_representation("TA", **{"num_states": num_states})

    tabular_features = np.vstack([TF[states[i]] for i in range(num_states)])
    assert np.array_equiv(
        tabular_features - np.eye(num_states), np.zeros((num_states, num_states))
    )


@pytest.mark.parametrize("num_states", [5, 19])
def test_dependent_features_all_columns_sum_up_to_in_features(num_states):
    states = np.arange(num_states).reshape(-1, 1)
    DF = get_representation(
        "D",
        unit_norm=False,
        **{"num_states": num_states, "num_dims": num_states // 2 + 1}
    )
    dependent_features = np.vstack([DF[states[i]] for i in range(num_states)])
    assert np.array_equiv(
        np.sum(dependent_features, axis=0),
        np.ones(num_states // 2 + 1) * num_states // 2 + 1,
    )


@pytest.mark.parametrize("num_states", [5, 19])
def test_random_features_binary_num_ones(num_states):
    states = np.arange(num_states).reshape(-1, 1)
    num_ones = num_states // 4 + 1
    RF = get_representation(
        "RB",
        **{
            "num_states": num_states,
            "seed": 0,
            "num_features": num_states // 2 + 1,
            "num_ones": num_ones,
        }
    )

    random_features = np.vstack([RF[states[i]] for i in range(num_states)])
    for i in range(num_states):
        assert np.nonzero(random_features[i][0] == num_ones)


@pytest.mark.parametrize("num_states, order, out_features", [(5, 3, 4), (19, 9, 10)])
def test_fourier_features_num_features(num_states, order, out_features):
    states = np.arange(num_states).reshape(-1, 1)

    BF = get_representation("F", **{"order": order, "num_dims": 1})

    features = np.vstack([BF[states[i]] for i in range(num_states)])
    assert features.shape[1] == out_features


def test_same_random_feature_for_state_in_an_episode():
    states = np.array([1, 1, 2, 2, 2, 3, 3]).reshape(-1, 1)

    RF = get_representation(
        "RB", **{"num_states": 7, "seed": 6, "num_features": 7 // 2, "num_ones": 2}
    )

    random_features = np.vstack([RF[states[i]] for i in range(7)])
    assert np.array_equiv(random_features[0], random_features[1])

    assert np.array_equiv(random_features[2], random_features[3])
    assert np.array_equiv(random_features[2], random_features[4])
    assert np.array_equiv(random_features[3], random_features[4])

    assert np.array_equiv(random_features[5], random_features[6])


@pytest.mark.parametrize("num_states", [5])
def test_step_size_fourier_cosine_features(num_states):
    step_size = 0.5

    F = get_representation(
        "F",
        **{
            "min_x": 0,
            "max_x": num_states - 1,
            "a": 0,
            "b": 1,
            "num_dims": 1,
            "order": 3,
        }
    )

    new_step_size = per_feature_step_size_fourier_KOT(step_size, F.num_features, F.C)

    assert np.array_equiv(
        new_step_size, np.array([step_size, step_size, step_size / 2, step_size / 3])
    )

    F = get_representation(
        "F",
        **{
            "min_x": 0,
            "max_x": num_states - 1,
            "a": 0,
            "b": 1,
            "num_dims": 2,
            "order": 2,
        }
    )
    new_step_size = per_feature_step_size_fourier_KOT(step_size, F.num_features, F.C)

    assert np.array_equiv(
        new_step_size,
        np.array(
            [
                step_size,
                step_size,
                step_size / 2,
                step_size,
                step_size / np.sqrt(2),
                step_size / np.sqrt(5),
                step_size / 2,
                step_size / np.sqrt(5),
                step_size / np.sqrt(8),
            ]
        ),
    )
