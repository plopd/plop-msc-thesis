import numpy as np
import pytest

from utils.utils import get_feature


@pytest.mark.parametrize("num_states", [5, 19])
def test_tabular_features(num_states):
    states = np.arange(1, num_states + 1).reshape(-1, 1)
    tabular_features = np.vstack(
        [
            get_feature(states[i], **{"features": "tabular", "N": num_states})
            for i in range(num_states)
        ]
    )
    assert np.allclose(np.sum(tabular_features, axis=1), np.ones(num_states))


@pytest.mark.parametrize("num_states", [5, 19])
def test_inverted_features_is_unit_norm(num_states):
    states = np.arange(1, num_states + 1).reshape(-1, 1)
    inverted_features = np.vstack(
        [
            get_feature(states[i], **{"features": "inverted", "N": num_states})
            for i in range(num_states)
        ]
    )
    norm_inv_features = np.linalg.norm(inverted_features, axis=1)
    assert np.allclose(norm_inv_features, np.ones(num_states))


@pytest.mark.parametrize("num_states", [5, 19])
def test_dependent_features_is_unit_norm(num_states):
    states = np.arange(1, num_states + 1).reshape(-1, 1)
    dependent_features = np.vstack(
        [
            get_feature(states[i], **{"features": "inverted", "N": num_states})
            for i in range(num_states)
        ]
    )
    norm_dep_features = np.linalg.norm(dependent_features, axis=1)
    assert np.allclose(norm_dep_features, np.ones(num_states))


@pytest.mark.parametrize("num_states", [5, 19])
def test_random_features_binary_num_ones(num_states):
    states = np.arange(1, num_states + 1).reshape(-1, 1)
    num_ones = num_states // 4 + 1
    random_features = np.vstack(
        [
            get_feature(
                states[i],
                normalize=False,
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
    assert np.allclose(np.sum(random_features, axis=1), np.ones(num_states) * num_ones)


@pytest.mark.parametrize("num_states", [5, 19])
def test_random_features_nonbinary_is_unit_norm(num_states):
    states = np.arange(1, num_states + 1).reshape(-1, 1)
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
    states = np.arange(1, num_states + 1).reshape(-1, 1)
    random_features = np.vstack(
        [
            get_feature(states[i], **{"features": "fourier", "order": order})
            for i in range(num_states)
        ]
    )
    assert random_features.shape[1] == out_features
