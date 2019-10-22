#!/bin/bash

#SBATCH --account=def-sutton
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --job-name slurm_submit.sh
#SBATCH --output=/home/plopd/scratch/chain/output/submit_%a.txt
#SBATCH --error=/home/plopd/scratch/chain/error/submit_%a.txt

export OMP_NUM_THREADS=1

source /home/plopd/projects/def-sutton/plopd/plop-msc-thesis-venv/bin/activate

echo "${SLURM_ARRAY_TASK_ID}"
echo "Current working directory is $(pwd)"
echo "Running on hostname $(hostname)"
echo "Starting run at: $(date)"

python -m scripts.run_chain_experiment "${SLURM_ARRAY_TASK_ID}"
echo "Finished with exit code $? at: $(date)"
