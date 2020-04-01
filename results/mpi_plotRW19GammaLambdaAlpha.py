import sys

from mpi4py import MPI

from results.old_random_walk_alpha_lambda_gamma import plot


def main():
    COMM = MPI.COMM_WORLD
    plot(sweep_id=COMM.Get_rank(), config_fn=sys.argv[1])


if __name__ == "__main__":
    main()
