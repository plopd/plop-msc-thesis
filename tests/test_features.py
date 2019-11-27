import numpy as np
import pytest

from utils.utils import get_feature


@pytest.mark.parametrize("num_states", [5, 19])
def test_tabular_features(num_states):
    states = np.arange(num_states).reshape(-1, 1)
    tabular_features = np.vstack(
        [
            get_feature(states[i], **{"features": "tabular", "in_features": num_states})
            for i in range(num_states)
        ]
    )
    assert np.allclose(np.sum(tabular_features, axis=1), np.ones(num_states))


@pytest.mark.parametrize("num_states", [5, 19])
def test_inverted_features_is_unit_norm(num_states):
    states = np.arange(num_states).reshape(-1, 1)
    inverted_features = np.vstack(
        [
            get_feature(
                states[i], **{"features": "inverted", "in_features": num_states}
            )
            for i in range(num_states)
        ]
    )
    norm_inv_features = np.linalg.norm(inverted_features, axis=1)
    assert np.allclose(norm_inv_features, np.ones(num_states))


@pytest.mark.parametrize("num_states", [5, 19])
def test_inverted_features_all_columns_sum_up_to_in_features_minus_one(num_states):
    states = np.arange(num_states).reshape(-1, 1)
    dependent_features = np.vstack(
        [
            get_feature(
                states[i],
                **{"features": "inverted", "in_features": num_states},
                unit_norm=False
            )
            for i in range(num_states)
        ]
    )
    assert np.allclose(
        np.sum(dependent_features, axis=0), np.ones(num_states) * (num_states - 1)
    )


@pytest.mark.parametrize("num_states", [5, 19])
def test_dependent_features_is_unit_norm(num_states):
    states = np.arange(num_states).reshape(-1, 1)
    dependent_features = np.vstack(
        [
            get_feature(
                states[i],
                **{"features": "dependent", "in_features": num_states // 2 + 1}
            )
            for i in range(num_states)
        ]
    )
    norm_dep_features = np.linalg.norm(dependent_features, axis=1)
    assert np.allclose(norm_dep_features, np.ones(num_states))


@pytest.mark.parametrize("num_states", [5, 19])
def test_dependent_features_all_columns_sum_up_to_in_features(num_states):
    states = np.arange(num_states).reshape(-1, 1)
    in_features = num_states // 2 + 1
    dependent_features = np.vstack(
        [
            get_feature(
                states[i],
                **{"features": "dependent", "in_features": in_features},
                unit_norm=False
            )
            for i in range(num_states)
        ]
    )
    assert np.allclose(
        np.sum(dependent_features, axis=0), np.ones(in_features) * in_features
    )


@pytest.mark.parametrize("num_states", [5, 19])
def test_random_features_binary_num_ones(num_states):
    states = np.arange(num_states).reshape(-1, 1)
    num_ones = num_states // 4 + 1
    random_features = np.vstack(
        [
            get_feature(
                states[i],
                **{
                    "features": "random-binary",
                    "N": num_states,
                    "seed": 0,
                    "in_features": num_states // 2 + 1,
                    "num_ones": num_ones,
                }
            )
            for i in range(num_states)
        ]
    )
    for i in range(num_states):
        assert np.nonzero(random_features[i][0] == num_ones)


@pytest.mark.parametrize("num_states", [5, 19])
def test_random_features_nonbinary_is_unit_norm(num_states):
    states = np.arange(num_states).reshape(-1, 1)
    random_features = np.vstack(
        [
            get_feature(
                states[i],
                **{
                    "features": "random-nonbinary",
                    "N": num_states,
                    "seed": 0,
                    "in_features": num_states // 2,
                }
            )
            for i in range(num_states)
        ]
    )
    norm_random_features = np.linalg.norm(random_features, axis=1)
    assert np.allclose(norm_random_features, np.ones(num_states))


@pytest.mark.parametrize("num_states, order, out_features", [(5, 3, 4), (19, 9, 10)])
def test_fourier_features_num_features(num_states, order, out_features):
    states = np.arange(num_states).reshape(-1, 1)
    in_features = (order + 1) ** states.shape[1]
    random_features = np.vstack(
        [
            get_feature(
                states[i],
                **{"features": "fourier", "order": order, "in_features": in_features}
            )
            for i in range(num_states)
        ]
    )
    assert random_features.shape[1] == out_features


def test_same_random_feature_for_state_in_an_episode():
    states = np.array([1, 1, 2, 2, 2, 3, 3]).reshape(-1, 1)
    random_features = np.vstack(
        [
            get_feature(
                states[i],
                **{
                    "features": "random-binary",
                    "N": 7,
                    "seed": 6,
                    "in_features": 7 // 2,
                    "num_ones": 2,
                }
            )
            for i in range(7)
        ]
    )
    assert np.allclose(random_features[0], random_features[1])

    assert np.allclose(random_features[2], random_features[3])
    assert np.allclose(random_features[2], random_features[4])
    assert np.allclose(random_features[3], random_features[4])

    assert np.allclose(random_features[5], random_features[6])
