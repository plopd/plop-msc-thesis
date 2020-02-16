import argparse

from alphaex.submitter import Submitter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_jobs", type=int, help="number of jobs to run", required=True
    )
    parser.add_argument("--experiment_rootdir", type=str, required=True)
    parser.add_argument("--config_filename", type=str, required=True)
    parser.add_argument("--python_module", type=str, default="scripts.run_experiment")
    parser.add_argument(
        "--script_path",
        type=str,
        help="script path for submitter",
        default="slurm_submit_jobs.sh",
    )

    parser.add_argument("--time", type=str, help="time [HH:MM:SS]", required=True)

    parser.add_argument("--account", type=str, default="def-sutton")

    parser.add_argument(
        "--mem_per_cpu", type=str, help="mem-per-cpu (e.g. G or MB)", default="1G"
    )

    args = parser.parse_args()

    rootdir = "/home/plopd/scratch"
    project_root_dir = "/home/plopd/projects/def-sutton/plopd/plop-msc-thesis"
    script_path = args.script_path
    output_path = f"{rootdir}/{args.experiment_rootdir}/{args.config_filename}/output"
    error_path = f"{rootdir}/{args.experiment_rootdir}/{args.config_filename}/error"

    clusters = [
        {
            "name": "mp2",
            "account": args.account,
            "capacity": 1000,
            "project_root_dir": project_root_dir,
            "exp_results_from": [output_path, error_path],
            "exp_results_to": [output_path, error_path],
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
        sbatch_params={
            "time": args.time,
            "mem-per-cpu": args.mem_per_cpu,
            "output": f"{output_path}/slurm-%A_%5a.txt",
            "error": f"{error_path}/slurm-%A_%5a.txt",
        },
        export_params={
            "python_module": args.python_module,
            "config_filename": args.config_filename,
        },
    )
    submitter.submit()


if __name__ == "__main__":
    main()
