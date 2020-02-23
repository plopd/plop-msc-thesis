import sys

from mpi4py import MPI

from results.plot_predictions_randomwalk import plot

if __name__ == "__main__":
    COMM = MPI.COMM_WORLD
    id = COMM.Get_rank()
    config = sys.argv[1]
    plot(id, config)
