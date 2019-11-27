#!/bin/bash

#SBATCH --account=def-sutton
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --output=/home/plopd/scratch/output/slurm-%A_%5a.txt
#SBATCH --error=/home/plopd/scratch/error/slurm-%A_%5a.txt

export OMP_NUM_THREADS=1

source /home/plopd/projects/def-sutton/plopd/plop-msc-thesis-venv/bin/activate

echo "Current working directory is $(pwd)."
echo "Running on hostname $(hostname)."

start=$(date +%s)
echo "Starting run at: $(date)."
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}" "CONFIG_FILE:" "${CONFIG_FILE}"
python -m scripts.run_chain_experiment "${SLURM_ARRAY_TASK_ID}" "${CONFIG_FILE}"
end=$(date +%s)
runtime=$((end-start))

echo "Finished with exit code $? at: $(date)."
echo "Executed in $runtime seconds."
