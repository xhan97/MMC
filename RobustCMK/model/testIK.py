import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math

torch.autograd.set_detect_anomaly(True)

from sklearn.datasets import load_iris


X, y = load_iris(return_X_y=True)
X = torch.tensor(X, dtype=torch.float32)


def check_for_nan(tensor, label):
    if torch.isnan(tensor).any():
        print(f"{label} contains NaN values")


def safe_softmax(X):
    C = torch.max(X, 1, True)[0]
    log_sum_exp = C + np.log(np.exp(X - C))
    return torch.exp(X - log_sum_exp)


# def iso_kernel(X, all_X, eta, psi):
#     map_tmp = None
#     if all_X is None:
#         all_X = X
#     np.random.seed(42)
#     samples_index = [
#         np.random.choice(len(all_X), psi, replace=False) for _ in range(100)
#     ]
#     for s_index in samples_index:
#         samples = all_X[s_index]
#         dist = -2*eta*torch.cdist(X, samples)
#         log_soft_max_dist = torch.clamp(F.log_softmax(dist, dim=1), min=-20, max=20)
#         soft_max_dist = torch.exp(log_soft_max_dist)
#         # soft_max_dist = safe_softmax(dist)
#         check_for_nan(soft_max_dist,"soft_max_dist")
#         soft_dist = torch.sqrt(soft_max_dist)

#         check_for_nan(soft_dist,"soft_dist")
#         if map_tmp is None:
#             map_tmp = soft_dist
#         else:
#             map_tmp = torch.hstack([map_tmp, soft_dist])
#         if torch.mm(soft_dist, soft_dist.T).isinf().any():
#             print(soft_dist)
#         if torch.mm(soft_dist, soft_dist.T).isnan().any():
#             print(soft_dist)
#     ik_similarity = torch.mm(map_tmp, map_tmp.T) / len(samples_index)
#     assert ik_similarity.shape == (X.shape[0], X.shape[0])
#     return ik_similarity


def iso_kernel(X, all_X, eta, psi, t=100):
    map_tmp = None
    if all_X is None:
        all_X = X
    samples_index = np.array(
        [np.random.choice(len(all_X), psi, replace=False) for _ in range(t)]
    )
    unique_samples_index, indices = np.unique(
        np.concatenate(samples_index), return_inverse=True
    )
    samples_index_inverse = np.split(indices, t)
    scaling_factor = math.sqrt(X.shape[1])
    sim_unique_samples = torch.mm(all_X[unique_samples_index], X.T)  # / scaling_factor
    for index_inverse in samples_index_inverse:
        log_soft_max_sim = torch.clamp(
            F.log_softmax(eta * sim_unique_samples[index_inverse], dim=0) / 2,
            min=-20,
            max=20,
        )
        soft_max_sim = torch.exp(log_soft_max_sim)
        if map_tmp is None:
            map_tmp = soft_max_sim
        else:
            map_tmp = torch.vstack([map_tmp, soft_max_sim])
    ik_similarity = torch.mm(map_tmp.T, map_tmp) / t
    assert ik_similarity.shape == (X.shape[0], X.shape[0])
    return ik_similarity


Xs = torch.tensor(X, dtype=torch.float32, requires_grad=True)
x_map = iso_kernel(X=Xs, all_X=None, eta=1, psi=8)
x_map.sum().backward()
print(Xs.grad)
