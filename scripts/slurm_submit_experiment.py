#######################################################################
# Copyright (C) 2019 Yi Wan(wan6@ualberta.ca)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import argparse

from alphaex.submitter import Submitter


def main():
    parser = argparse.ArgumentParser(description="Submit experiment to slurm clusters.")
    parser.add_argument(
        "-num_jobs", type=int, help="number of jobs to run", required=True
    )
    parser.add_argument(
        "-config_file", type=str, default=None, help="name of config file"
    )
    parser.add_argument(
        "-script_path",
        type=str,
        help="script path for submitter",
        default="slurm_submit.sh",
    )
    args = parser.parse_args()

    num_jobs = args.num_jobs
    config_file = args.config_file
    experiment_root_dir = "/home/plopd/scratch"
    project_root_dir = "/home/plopd/projects/def-sutton/plopd/plop-msc-thesis"
    script_path = args.script_path

    clusters = [
        {
            "name": "mp2",
            "capacity": 1000,
            "project_root_dir": project_root_dir,
            "exp_results_from": [
                f"{experiment_root_dir}/output",
                f"{experiment_root_dir}/error",
            ],
            "exp_results_to": [
                f"{experiment_root_dir}/output",
                f"{experiment_root_dir}/error",
            ],
        }
        # {
        #     "name": "cedar",
        #     "capacity": 1000,
        #     "project_root_dir": project_root_dir,
        #     "exp_results_from": [
        #         f"{experiment_root_dir}/output",
        #         f"{experiment_root_dir}/error",
        #     ],
        #     "exp_results_to": [
        #         f"{experiment_root_dir}/output",
        #         f"{experiment_root_dir}/error",
        #     ],
        # },
    ]
    submitter = Submitter(clusters, num_jobs, script_path, config_file)
    submitter.submit()


if __name__ == "__main__":
    main()
