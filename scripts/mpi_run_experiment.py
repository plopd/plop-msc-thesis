import sys

from mpi4py import MPI

from scripts.run_experiment import run
from utils.decorators import timer


@timer
def main():
    COMM = MPI.COMM_WORLD
    run(sweep_id=COMM.Get_rank(), config_fn=sys.argv[1])


if __name__ == "__main__":
    main()
