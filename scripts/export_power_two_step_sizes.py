import sys

import numpy as np

from utils.utils import set_emphatic_step_size

low = int(sys.argv[1])
high = int(sys.argv[2])
step = int(sys.argv[3])
interest = int(sys.argv[4])
gamma = float(sys.argv[5])
lmbda = float(sys.argv[6])

step_sizes_td = [0.1 * 2 ** i for i in range(low, high, step)]

print("TD: ", len(step_sizes_td), step_sizes_td, end="\n")

step_sizes_etd = [
    np.float32(set_emphatic_step_size(sz, interest, gamma, lmbda))
    for sz in step_sizes_td
]

print("ETD: ", len(step_sizes_etd), step_sizes_etd, end="\n")
