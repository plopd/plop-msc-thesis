#!/bin/bash

#SBATCH --account=def-sutton
#SBATCH --time=00:55:00
#SBATCH --mem-per-cpu=1G
#SBATCH --output=/home/plopd/scratch/output/slurm-%A_%5a.txt
#SBATCH --error=/home/plopd/scratch/error/slurm-%A_%5a.txt

export OMP_NUM_THREADS=1

source /home/plopd/projects/def-sutton/plopd/plop-msc-thesis-venv/bin/activate

echo "Current working directory is $(pwd)."
echo "Running on hostname $(hostname)."

start=$(date +%s)
echo "Starting run at: $(date)."
SWEEP_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${missing_values}")

python -m "${python_module}" "${SWEEP_ID}" "${config_file}"
end=$(date +%s)
runtime=$((end-start))

echo "Finished with exit code $? at: $(date)."
echo "Executed in $runtime seconds."
