import sys

import utils.utils as utils


def main():
    cfg_experiment = sys.argv[1]
    cfg_state_distribution = sys.argv[2]
    cfg_true_v = sys.argv[3]

    utils.calculate_state_distribution(cfg_state_distribution)
    utils.calculate_true_v(cfg_true_v)
    utils.export_params_from_config_random_walk(cfg_experiment)


if __name__ == "__main__":
    main()
