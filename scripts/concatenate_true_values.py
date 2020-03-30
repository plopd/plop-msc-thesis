import os
import sys
from pathlib import Path

import numpy as np
from alphaex.sweeper import Sweeper

from utils.utils import path_exists


def main():
    concatenate_true_values(config_fn=sys.argv[1])


def concatenate_true_values(config_fn):
    config_root_path = Path(__file__).parents[1] / "configs"
    sweeper = Sweeper(config_root_path / f"{config_fn}.json")
    config = sweeper.parse(0)
    save_rootpath = Path(f"{os.environ.get('SCRATCH')}") / f"{config.get('problem')}"
    save_rootpath = path_exists(save_rootpath)
    discount_rate = config.get("discount_rate")
    num_obs = config.get("num_obs")
    true_values = []
    for i in range(num_obs):
        filename = f"{i}-discount_rate_{discount_rate}".replace(".", "_")
        arr = np.load(f"{filename}.npy")
        true_values.append(arr)

    true_values = np.hstack(true_values)
    np.save(
        save_rootpath / f"true_values-discount_rate_{discount_rate}".replace(".", "_"),
        true_values,
    )


if __name__ == "__main__":
    main()
