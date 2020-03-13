import json

import numpy as np

lambdas = [0.0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
gammas = [0.99, 0.9995]
alphas = [
    [
        np.arange(0, 1.8, 0.1),
        np.arange(0, 1.8, 0.1),
        np.arange(0, 0.99, 0.09),
        np.arange(0, 0.65, 0.05),
        np.arange(0, 0.53, 0.03),
        np.arange(0, 0.32, 0.02),
        np.arange(0, 0.11, 0.01),
        np.arange(0, 0.044, 0.004),
    ],
    [
        np.arange(0, 1.8, 0.1),
        np.arange(0, 1.8, 0.1),
        np.arange(0, 1.29, 0.09),
        np.arange(0, 1.15, 0.05),
        np.arange(0, 1.13, 0.03),
        np.arange(0, 1.32, 0.02),
        np.arange(0, 1.11, 0.01),
        np.arange(0, 0.044, 0.004),
    ],
]


lmbda2alpha = {}
for g, gamma in enumerate(gammas):
    lmbda2alpha[gamma] = {"discount_rate": [gamma], "lambdas_step_sizes": []}
    for l, lmbda in enumerate(lambdas):
        szs = alphas[g][l].tolist()
        szs = [float(f"{sz:{4}.{3}f}") for sz in szs]
        lmbda2alpha[gamma]["lambdas_step_sizes"].append(
            {"trace_decay": [lmbda], "step_size": szs}
        )
print(json.dumps(lmbda2alpha, indent=4))
