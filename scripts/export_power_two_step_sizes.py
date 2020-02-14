import argparse

import numpy as np

from utils.utils import emphatic_multiplier_step_size


def main():
    """
    > python -m scripts.export_power_two_step_sizes --range -6 2 2 --interest 1 --discount_factor 0.99 --trace_decay 0.0
    Returns:

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--range", nargs="+", type=int)
    parser.add_argument("--interest", type=int)
    parser.add_argument("--discount_rate", type=float)
    parser.add_argument("--trace_decay", type=float)
    parser.add_argument("--tilings", type=int, default=1)

    args = parser.parse_args()

    step_sizes_td = [0.1 * 2 ** i for i in range(*args.range)]

    factor = emphatic_multiplier_step_size(
        args.interest, args.discount_rate, args.trace_decay
    )
    step_sizes_etd = [np.float32(factor * sz) for sz in step_sizes_td]

    return step_sizes_td, step_sizes_etd


if __name__ == "__main__":
    step_sizes_td, step_sizes_etd = main()
    print(
        f"TD: {len(step_sizes_td)}, {step_sizes_td},\n"
        f"ETD: {len(step_sizes_etd)}, {step_sizes_etd}"
    )
