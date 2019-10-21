import numpy as np
from utils.utils import get_dependent_features
from utils.utils import get_inverted_features

N = 19
inverted_features = get_inverted_features(N)
norm_inv_features = np.linalg.norm(inverted_features, axis=1)
print(np.allclose(norm_inv_features, np.ones(N)))

dependent_features = get_dependent_features(N)
norm_dep_features = np.linalg.norm(dependent_features, axis=1)
print(np.allclose(norm_dep_features, np.ones(N)))
