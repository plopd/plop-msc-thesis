#!/bin/bash

#SBATCH --account=def-sutton
#SBATCH --time=00:35:00
#SBATCH --mem-per-cpu=1G
#SBATCH --job-name plot.sh
#SBATCH --output=/home/plopd/scratch/output/slurm-%A_%5a.txt
#SBATCH --error=/home/plopd/scratch/error/slurm-%A_%5a.txt

source /home/plopd/projects/def-sutton/plopd/plop-msc-thesis-venv/bin/activate

echo "Current working directory is $(pwd)."
echo "Running on hostname $(hostname)."

start=$(date +%s)
echo "Starting run at: $(date)."

python -m results.plot_predictions_randomwalk --num_states 5 --num_runs 100 --representations D --metric end --env Chain --experiment ChainDependent --discount_rate 0.99 --trace_decay 0.0 --baseline 1
end=$(date +%s)
runtime=$((end-start))

echo "Finished with exit code $? at: $(date)."
echo "Executed in $runtime seconds."
