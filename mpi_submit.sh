#!/bin/bash

NTASKS=$1

mpiexec -n $NTASKS python -m scripts.run_chain_experiment_mpi
