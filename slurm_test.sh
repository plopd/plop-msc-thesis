#!/bin/bash

#SBATCH --account=def-sutton
#SBATCH --time=00:01:00
#SBATCH --mem-per-cpu=10M
#SBATCH --job-name slurm_test.sh
#SBATCH --output=/home/plopd/scratch/test/output_%a.txt
#SBATCH --error=/home/plopd/scratch/test/error_%a.txt

export OMP_NUM_THREADS=1

echo "${SLURM_ARRAY_TASK_ID}"
echo "Current working directory is $(pwd)"
echo "Running on hostname $(hostname)"
echo "Starting run at: $(date)"

echo "Finished with exit code $? at: $(date)"
