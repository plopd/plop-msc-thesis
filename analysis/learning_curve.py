from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


data_path = Path(__file__).parents[3] / "data-plop-msc-thesis"
experiment_name = "chain_five"
idx = 79
data_file_name = f"{idx}_error.npy"
data = np.load(f"{data_path}/{experiment_name}/{data_file_name}", allow_pickle=True)

plt.plot(data)
plt.show()
