from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from analysis.learning_curve import get_LCA
from analysis.results import Result
from analysis.sensitivity_curve import get_SSA
from analysis.waterfall import get_WF
from utils.utils import path_exists

matplotlib.rcParams.update({"font.size": 18})

N = [5, 19]
FEATURES = ["tabular", "dependent", "random-binary", "random-nonbinary"]
INTEREST = ["uniform"]
DATA_PATH = "/home/plopd/scratch"
RUNS = 100
CUTOFF = 0.45
RESULTS_PATH = f"{Path(__file__)}/Chain"

path_exists(RESULTS_PATH)

for n in N:
    for f in FEATURES:
        for i in INTEREST:
            EXPERIMENT_NAME = f"Chain{n}TabularDependentRandomBinaryRandomNonbinary"
            N_RUNS_EXACT = (
                RUNS if i == "random-binary" or i == "random-nonbinary" else 1
            )

            result = Result(
                f"{EXPERIMENT_NAME}.json", DATA_PATH, EXPERIMENT_NAME, runs=RUNS
            )

            fig, axs = plt.subplots(1, 3, figsize=(20, 5), sharey="row", dpi=80)
            get_LCA(axs[0], result, n, f, i, N_RUNS_EXACT)
            get_SSA(axs[1], result, f, i, cutoff=CUTOFF)
            get_WF(axs[2], result, f, i, cutoff=CUTOFF)
            plt.tight_layout()
            plt.savefig(f"{RESULTS_PATH}/Chain_{n}_{f}_{i}")
