import argparse

from alphaex.submitter import Submitter


def main():
    parser = argparse.ArgumentParser(description="Submit experiment to slurm clusters.")
    parser.add_argument(
        "-num_jobs", type=int, help="number of jobs to run", required=True
    )
    parser.add_argument(
        "-config_file", type=str, required=True, help="name of config file"
    )
    parser.add_argument(
        "-python_module",
        type=str,
        default="scripts.run_experiment",
        help="python module to execute",
    )
    parser.add_argument(
        "-script_path",
        type=str,
        help="script path for submitter",
        default="slurm_submit_jobs.sh",
    )
    args = parser.parse_args()

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
    submitter = Submitter(
        clusters,
        args.num_jobs,
        script_path,
        export_params={
            "python_module": args.python_module,
            "config_file": args.config_file,
        },
    )
    submitter.submit()


if __name__ == "__main__":
    main()
