#!/bin/bash
#SBATCH --account=def-sutton
#SBATCH --error=../../logs/slurm-%j-%n-%a.err
#SBATCH --output=../../logs/slurm-%j-%n-%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

module load python/3.6
source /home/plopd/home/plopd/projects/def-sutton/plopd/plop-msc-thesis/venv-thesis-code/bin/activate

`sed -n "${SLURM_ARRAY_TASK_ID}p" < $1`

echo ${SLURM_ARRAY_TASK_ID}
echo "Current working directory is `pwd`"
echo "Running on hostname `hostname`"
echo "Starting run at: `date`"

python -m experiments.experiment_random_walk --agent ${agent} --n ${n} --phi ${phi} --alpha ${alpha} --gamma ${gamma} --lmbda ${lmbda} --env ${env} --N ${N} --s0 ${s0} --s_left_term ${s_left_term} --s_right_term ${s_right_term} --r_left ${r_left} --r_right ${r_right} --exp_name ${exp_name} --n_runs ${n_runs} --n_episodes ${n_episodes} --episode_eval_freq ${episode_eval_freq}
echo "Finished with exit code $? at: `date`"