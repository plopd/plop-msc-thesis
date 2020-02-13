import argparse


def main():
    """
    > python -m scripts.export_power_two_step_sizes --range -6 2 2 --interest 1 --discount_factor 0.99 --trace_decay 0.0
    Returns:

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--range", nargs="+", type=int)
    parser.add_argument("--interest", type=int)
    parser.add_argument("--discount_factor", type=float)
    parser.add_argument("--trace_decay", type=float)

    args = parser.parse_args()

    step_sizes_td = [0.1 * 2 ** i for i in range(*args.range)]

    # step_sizes_etd = [
    #     np.float32(
    #         set_emphatic_step_size(
    #             sz, args.interest, args.discount_factor, args.trace_decay
    #         )
    #     )
    #     for sz in step_sizes_td
    # ]

    return step_sizes_td


if __name__ == "__main__":
    step_sizes_td = main()
    print(f"TD: {len(step_sizes_td)}, {step_sizes_td}")
