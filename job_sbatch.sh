#!/bin/bash

source $HOME/projects/def-sutton/plopd/plop-msc-thesis-venv/bin/activate

echo "Current working directory is $(pwd)."
echo "Running on hostname $(hostname)."

start=$(date +%s)
echo "Starting run at: $(date)."

echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}" "config_filename:" "${config_filename}"
python -m "${python_module}" "${SLURM_ARRAY_TASK_ID}" "${config_filename}"
end=$(date +%s)
runtime=$((end-start))

echo "Finished with exit code $? at: $(date)."
echo "Executed in $runtime seconds."
