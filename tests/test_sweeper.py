from pathlib import Path

import pytest
from alphaex.sweeper import Sweeper

from analysis.results import Result


def test_sweeper_same_keys_for_all_experiments():
    cfg_filename, num_runs = "Test_Sweeper.json", 100
    sweeper = Sweeper(f"{Path(__file__).parents[1]}/configs/{cfg_filename}")
    src_dct = {
        "run": 0,
        "n_episodes": 50000,
        "max_episode_steps": 0,
        "episode_eval_freq": 1,
        "output_dir": "~/scratch/Chain/ChainTabularDependent",
        "env": "chain",
        "N": 19,
        "algorithm": "etd",
        "alpha": 1.192_092_895_507_812_5e-07,
        "gamma": 1.0,
        "lmbda": 0.0,
        "features": "dependent",
        "in_features": 10,
        "interest": "uniform",
    }
    for sweep_id in range(sweeper.total_combinations * num_runs):
        rtn_dict = sweeper.parse(sweep_id)

        assert set(src_dct.keys()) == set(rtn_dict.keys())


@pytest.mark.parametrize(
    "name, features",
    [("td", "tabular"), ("td", "dependent"), ("etd", "tabular"), ("etd", "dependent")],
)
def test_sweeper_number_of_stepsizes_for_method(name, features):
    cfg_filename, num_runs = "Test_Sweeper", 100
    result = Result(f"{cfg_filename}.json", None, "Test_Sweeper", num_runs)

    param_vals = result.get_param_val(
        "alpha",
        {
            "algorithm": name,
            "env": "chain",
            "features": features,
            "interest": "uniform",
            "N": 19,
        },
    )

    assert len(param_vals) == 12


@pytest.mark.parametrize(
    "name, features",
    [("td", "tabular"), ("td", "dependent"), ("etd", "tabular"), ("etd", "dependent")],
)
def test_sweeper_number_of_experiments_for_learner_and_features(name, features):
    cfg_filename, num_runs = "Test_Sweeper", 100
    sweeper = Sweeper(f"{Path(__file__).parents[1]}/configs/{cfg_filename}.json")

    search_lst = sweeper.search(
        {
            "algorithm": name,
            "env": "chain",
            "features": features,
            "interest": "uniform",
            "N": 19,
        },
        num_runs,
    )

    assert len(search_lst) == 12

    for lst in search_lst:
        assert len(lst["ids"]) == 100


@pytest.mark.parametrize("name", ["td", "etd"])
def test_sweeper_number_of_experiments_for_learner(name):
    cfg_filename, num_runs = "Test_Sweeper", 100
    sweeper = Sweeper(f"{Path(__file__).parents[1]}/configs/{cfg_filename}.json")

    search_lst = sweeper.search(
        {"algorithm": name, "env": "chain", "interest": "uniform", "N": 19}, num_runs
    )

    assert len(search_lst) == 24

    for lst in search_lst:
        assert len(lst["ids"]) == 100
