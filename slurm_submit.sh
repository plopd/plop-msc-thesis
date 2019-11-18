#!/bin/bash

#SBATCH --account=def-sutton
#SBATCH --time=00:50:00
#SBATCH --mem-per-cpu=1G
#SBATCH --job-name slurm_submit.sh
#SBATCH --output=/home/plopd/scratch/output/slurm-%A_%5a.txt
#SBATCH --error=/home/plopd/scratch/error/slurm-%A_%5a.txt

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /home/plopd/projects/def-sutton/plopd/plop-msc-thesis-venv/bin/activate

echo "${SLURM_ARRAY_TASK_ID}"
echo "Current working directory is $(pwd)."
echo "Running on hostname $(hostname)."

start=`date +%s`
echo "Starting run at: $(date)."
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
python -m scripts.run_chain_experiment "${SLURM_ARRAY_TASK_ID}"
end=`date +%s`
runtime=$((end-start))
echo "Finished with exit code $? at: $(date)."
echo "Executed in $runtime seconds."
