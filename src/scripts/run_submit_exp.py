#######################################################################
# Copyright (C) 2019 Yi Wan(wan6@ualberta.ca)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
from alphaex.submitter import Submitter


def run_submit():
    clusters = [
        {
            "name": "mp2b",
            "capacity": 500,
            "project_root_dir": "/home/plopd/plop-msc-thesis/src",
            "exp_results_from": [
                "/home/plopd/scratch/chain_five/output",
                "/home/plopd/scratch/chain_five/error",
            ],
            "exp_results_to": [
                "/home/plopd/scratch/chain_five/output",
                "/home/plopd/scratch/chain_five/error",
            ],
        }
    ]
    num_jobs = 84  # one job = one param setting with one single run
    # repo_url = None  # git repo of experiment code
    script_path = "/home/plopd/plop-msc-thesis/src/submit_slurm.sh"
    submitter = Submitter(clusters, num_jobs, script_path)
    submitter.submit()


if __name__ == "__main__":
    run_submit()
