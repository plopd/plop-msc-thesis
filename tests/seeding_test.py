import numpy as np

from scripts.run_experiment import run


def test_same_episodes_across_algorithm_instances():
    experiments = []
    for i in range(8):
        experiments.append(run(i, "MountainCar_test"))

    assert np.array_equiv(
        np.array(experiments[2].episodes[0][0]), np.array(experiments[3].episodes[0][0])
    )
    assert np.array_equiv(
        np.array(experiments[2].episodes[0][1]), np.array(experiments[3].episodes[0][1])
    )
    assert np.array_equiv(
        np.array(experiments[6].episodes[0][0]), np.array(experiments[7].episodes[0][0])
    )
    assert np.array_equiv(
        np.array(experiments[6].episodes[0][1]), np.array(experiments[7].episodes[0][1])
    )
