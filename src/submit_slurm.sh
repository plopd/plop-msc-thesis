#!/bin/bash

#SBATCH --account=def-sutton
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=500M
#SBATCH --job-name submit_slurm.sh
#SBATCH --output=/home/plopd/scratch/chain_five/output/submit_%a.txt
#SBATCH --error=/home/plopd/scratch/chain_five/error/submit_%a.txt

export OMP_NUM_THREADS=1

source /home/plopd/plop-msc-thesis-venv/bin/activate

echo "${SLURM_ARRAY_TASK_ID}"
echo "Current working directory is $(pwd)"
echo "Running on hostname $(hostname)"
echo "Starting run at: $(date)"

python -m scripts.run_chain_exp "${SLURM_ARRAY_TASK_ID}"
echo "Finished with exit code $? at: $(date)"
