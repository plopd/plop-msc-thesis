import numpy as np
import pytest

from utils.utils import MSVE


@pytest.mark.parametrize("N, expected_res", [(5, 11.0), (19, 130.0)])
def test_MSVE_uniform_interest_and_state_distribution(N, expected_res):
    res = MSVE(np.arange(1, N + 1), np.zeros(N), np.ones(N) / N, np.ones(N))
    assert np.isclose(res, np.sqrt(expected_res))


@pytest.mark.parametrize("N, expected_res", [(5, 25.0), (19, 361.0)])
def test_MSVE_last_state_interest_and_state_distribution(N, expected_res):
    i = np.zeros(N)
    i[-1] = 1
    res = MSVE(np.arange(1, N + 1), np.zeros(N), np.ones(N) / N, i)
    assert np.isclose(res, np.sqrt(expected_res))
