import sys

from mpi4py import MPI

from results.randomwalk_LCA_SSA import plot


def main():
    COMM = MPI.COMM_WORLD
    plot(sweep_id=COMM.Get_rank(), config_fn=sys.argv[1])


if __name__ == "__main__":
    main()
