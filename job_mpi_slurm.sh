#!/bin/bash

#SBATCH --account=def-sutton
#SBATCH --time=00:50:00
#SBATCH --mem-per-cpu=3G
#SBATCH --ntasks=10

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /home/plopd/projects/def-sutton/plopd/plop-msc-thesis-venv/bin/activate

mpiexec -n $SLURM_NTASKS python -m scripts.run_chain_experiment_better
