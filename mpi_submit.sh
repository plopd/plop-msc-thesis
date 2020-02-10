#!/bin/bash

NTASKS=$1
CONFIG_FILE=$2

mpiexec -n $NTASKS python -m scripts.run_experiment_mpi $CONFIG_FILE
