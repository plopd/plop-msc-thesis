import argparse

from alphaex.submitter import Submitter


def main():
    parser = argparse.ArgumentParser(
        description="Submit experiments to slurm clusters."
    )
    parser.add_argument(
        "--num_jobs", type=int, help="number of jobs to run", required=True
    )
    parser.add_argument(
        "--config_file", type=str, required=True, help="name of config file"
    )
    parser.add_argument(
        "--python_module",
        type=str,
        default="scripts.run_experiment",
        help="python script to run",
    )
    parser.add_argument(
        "--script_path",
        type=str,
        help="script path for submitter",
        default="slurm_submit_jobs.sh",
    )

    parser.add_argument("--time", type=str, help="time [HH:MM:SS]", required=True)

    parser.add_argument(
        "--mem_per_cpu", type=str, help="mem-per-cpu (e.g. G or MB)", default="1G"
    )

    args = parser.parse_args()

    experiment_root_dir = "/home/plopd/scratch"
    project_root_dir = "/home/plopd/projects/def-sutton/plopd/plop-msc-thesis"
    script_path = args.script_path

    clusters = [
        {
            "name": "mp2",
            "account": "def-sutton",
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
        sbatch_params={"time": args.time, "mem-per-cpu": args.mem_per_cpu},
        export_params={
            "python_module": args.python_module,
            "config_file": args.config_file,
        },
    )
    submitter.submit()


if __name__ == "__main__":
    main()
