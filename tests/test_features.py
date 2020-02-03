import numpy as np
import pytest

from representations.representations import get_representation


@pytest.mark.parametrize("num_states", [5, 19])
def test_tabular_features(num_states):
    states = np.arange(num_states).reshape(-1, 1)
    TF = get_representation("TF", **{"N": num_states})

    tabular_features = np.vstack([TF[states[i]] for i in range(num_states)])
    assert np.allclose(np.sum(tabular_features, axis=1), np.ones(num_states))


# @pytest.mark.parametrize("num_states", [5, 19])
# def test_inverted_features_all_columns_sum_up_to_in_features_minus_one(num_states):
#     states = np.arange(num_states).reshape(-1, 1)
#     dependent_features = np.vstack(
#         [
#             get_feature(
#                 states[i],
#                 **{"representations": "inverted", "num_dims": num_states},
#                 unit_norm=False
#             )
#             for i in range(num_states)
#         ]
#     )
#     assert np.allclose(
#         np.sum(dependent_features, axis=0), np.ones(num_states) * (num_states - 1)
#     )


@pytest.mark.parametrize("num_states", [5, 19])
def test_dependent_features_all_columns_sum_up_to_in_features(num_states):
    states = np.arange(num_states).reshape(-1, 1)
    DF = get_representation(
        "DF", unit_norm=False, **{"N": num_states, "num_dims": num_states // 2 + 1}
    )
    dependent_features = np.vstack([DF[states[i]] for i in range(num_states)])
    assert np.allclose(
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
            "N": num_states,
            "seed": 0,
            "num_dims": num_states // 2 + 1,
            "num_ones": num_ones,
        }
    )

    random_features = np.vstack([RF[states[i]] for i in range(num_states)])
    for i in range(num_states):
        assert np.nonzero(random_features[i][0] == num_ones)


@pytest.mark.parametrize("num_states", [5, 19])
def test_random_features_nonbinary_is_unit_norm(num_states):
    states = np.arange(num_states).reshape(-1, 1)
    RF = get_representation(
        "RNB", **{"N": num_states, "seed": 0, "num_dims": num_states // 2}
    )

    random_features = np.vstack([RF[states[i]] for i in range(num_states)])
    norm_random_features = np.linalg.norm(random_features, axis=1)
    assert np.allclose(norm_random_features, np.ones(num_states))


@pytest.mark.parametrize("num_states, order, out_features", [(5, 3, 4), (19, 9, 10)])
def test_fourier_features_num_features(num_states, order, out_features):
    states = np.arange(num_states).reshape(-1, 1)

    BF = get_representation("F", **{"order": order, "num_dims": 1})

    features = np.vstack([BF[states[i]] for i in range(num_states)])
    assert features.shape[1] == out_features


def test_same_random_feature_for_state_in_an_episode():
    states = np.array([1, 1, 2, 2, 2, 3, 3]).reshape(-1, 1)

    RF = get_representation(
        "RB", **{"N": 7, "seed": 6, "num_dims": 7 // 2, "num_ones": 2}
    )

    random_features = np.vstack([RF[states[i]] for i in range(7)])
    assert np.allclose(random_features[0], random_features[1])

    assert np.allclose(random_features[2], random_features[3])
    assert np.allclose(random_features[2], random_features[4])
    assert np.allclose(random_features[3], random_features[4])

    assert np.allclose(random_features[5], random_features[6])
