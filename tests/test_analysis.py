import matplotlib
import matplotlib.pyplot as plt

from analysis.learning_curve import get_LCA
from analysis.results import Result
from analysis.sensitivity_curve import get_SSA
from analysis.waterfall import get_WF

matplotlib.rcParams.update({"font.size": 18})

N = 19
FEATURES = "tabular"
INTEREST = "uniform"
DATA_PATH = "/Users/saipiens/scratch"
RUNS = 100
CUTOFF = 0.45

EXPERIMENT_NAME = f"Chain{N}withUniformAndRandomInterest"
N_RUNS_EXACT = RUNS if INTEREST == "random binary" else 1

result = Result(f"{EXPERIMENT_NAME}.json", DATA_PATH, EXPERIMENT_NAME, runs=RUNS)

fig, axs = plt.subplots(1, 3, figsize=(20, 5), sharey="row", dpi=80)
get_LCA(axs[0], result, N, FEATURES, INTEREST, N_RUNS_EXACT)
get_SSA(axs[1], result, FEATURES, INTEREST, cutoff=CUTOFF)
get_WF(axs[2], result, FEATURES, INTEREST, cutoff=CUTOFF)
plt.yticks([0.45, 0.35, 0.25, 0.15, 0.05])
plt.tight_layout()
plt.show()
