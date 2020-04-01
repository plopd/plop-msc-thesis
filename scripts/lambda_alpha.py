import json

import numpy as np

lambdas = [0.0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
gammas = [0.99]
alphas = [
    [
        np.linspace(0, 2.0, 10, endpoint=False),
        np.linspace(0, 2.0, 10, endpoint=False),
        np.linspace(0, 0.95, 10, endpoint=False),
        np.linspace(0, 0.65, 10, endpoint=False),
        np.linspace(0, 0.55, 10, endpoint=False),
        np.linspace(0, 0.35, 10, endpoint=False),
        np.linspace(0, 0.25, 10, endpoint=False),
        np.linspace(0, 0.15, 10, endpoint=False),
    ]
]


lmbda2alpha = {}
for g, gamma in enumerate(gammas):
    lmbda2alpha[gamma] = {"discount_rate": [gamma], "lambdas_step_sizes": []}
    for l, lmbda in enumerate(lambdas):
        szs = alphas[g][l].tolist()
        szs = [float(f"{sz:{5}.{4}f}") for i, sz in enumerate(szs)]
        lmbda2alpha[gamma]["lambdas_step_sizes"].append(
            {"trace_decay": [lmbda], "step_size": szs}
        )
print(json.dumps(lmbda2alpha, indent=4))
