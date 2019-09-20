#!/bin/bash

python -m experiments.experiment_random_walk --agent TD --n 5 --phi tabular --alpha 0.125 --gamma 1.0 --lmbda 0.0 --env RandomWalkEnvironment --N 5 --s0 3 --s_left_term 0 --s_right_term 6 --r_left 0 --r_right 1 --exp_name smoke_test_five_states_random_walk --n_runs 2 --n_episodes 5 --episode_eval_freq 1