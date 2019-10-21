#######################################################################
# Copyright (C) 2019 Yi Wan(wan6@ualberta.ca)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
from alphaex.submitter import Submitter


def run_submit():

    exp_results_rootpath = "/home/plopd/scratch/chain19"
    project_root_dir = "/home/plopd/projects/def-sutton/plopd/plop-msc-thesis"

    clusters = [
        {
            "name": "mp2b",
            "capacity": 10,
            "project_root_dir": project_root_dir,
            "exp_results_from": [
                f"{exp_results_rootpath}/output",
                f"{exp_results_rootpath}/error",
            ],
            "exp_results_to": [
                f"{exp_results_rootpath}/output",
                f"{exp_results_rootpath}/error",
            ],
        },
        {
            "name": "cedar",
            "capacity": 10,
            "project_root_dir": project_root_dir,
            "exp_results_from": [
                f"{exp_results_rootpath}/output",
                f"{exp_results_rootpath}/error",
            ],
            "exp_results_to": [
                f"{exp_results_rootpath}/output",
                f"{exp_results_rootpath}/error",
            ],
        },
    ]
    num_jobs = 20  # =num_combinations x n_runs
    repo_url = (
        "https://github.com/plopd/plop-msc-thesis.git"
    )  # git repo of experiment code
    script_path = (
        "/home/plopd/projects/def-sutton/plopd/plop-msc-thesis/slurm_submit.sh"
    )
    submitter = Submitter(clusters, num_jobs, script_path, repo_url=repo_url)
    submitter.submit()


if __name__ == "__main__":
    run_submit()
