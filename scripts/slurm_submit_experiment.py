#######################################################################
# Copyright (C) 2019 Yi Wan(wan6@ualberta.ca)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import sys

from alphaex.submitter import Submitter


def run_submit(num_jobs):
    EXPERIMENT_ROOT_DIR = "/home/plopd/scratch"
    PROJECT_ROOT_DIR = "/home/plopd/projects/def-sutton/plopd/plop-msc-thesis"
    SCRIPT_PATH = "slurm_submit.sh"
    NUM_JOBS = num_jobs

    CLUSTERS = [
        {
            "name": "mp2b",
            "capacity": 1000,
            "project_root_dir": PROJECT_ROOT_DIR,
            "exp_results_from": [
                f"{EXPERIMENT_ROOT_DIR}/output",
                f"{EXPERIMENT_ROOT_DIR}/error",
            ],
            "exp_results_to": [
                f"{EXPERIMENT_ROOT_DIR}/output",
                f"{EXPERIMENT_ROOT_DIR}/error",
            ],
        }
        # {
        #     "name": "cedar",
        #     "capacity": 1000,
        #     "project_root_dir": PROJECT_ROOT_DIR,
        #     "exp_results_from": [
        #         f"{EXPERIMENT_ROOT_DIR}/output",
        #         f"{EXPERIMENT_ROOT_DIR}/error",
        #     ],
        #     "exp_results_to": [
        #         f"{EXPERIMENT_ROOT_DIR}/output",
        #         f"{EXPERIMENT_ROOT_DIR}/error",
        #     ],
        # },
    ]
    submitter = Submitter(
        clusters=CLUSTERS, total_num_jobs=NUM_JOBS, script_path=SCRIPT_PATH
    )
    submitter.submit()


if __name__ == "__main__":
    num_jobs = int(sys.argv[1])
    run_submit(num_jobs)
