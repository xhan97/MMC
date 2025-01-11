import torch
import numpy as np
import torch.nn.functional as F
import math

# seed = 2021
# torch.manual_seed(seed)


# pytorch version
# Gaussian exp(-||x - y||^2/(2 * sigma^2))
def gaussian_kernel(x, y=None, sigma=None):
    if sigma is None:
        gamma = 1 / x.shape[1]
    else:
        gamma = 1 / (2 * sigma**2)
    if y is None:
        y = x
    dist = torch.exp(-torch.norm(x[:, None] - y, p=2, dim=-1) ** 2 * gamma)
    return dist


def rbf_kernel(x, y=None, gamma=None):
    if gamma is None:
        gamma = 1 / x.shape[1]
    if y is None:
        y = x
    dist = torch.exp(-torch.norm(x[:, None] - y, p=2, dim=-1) ** 2 * gamma)
    # dist = torch.sum((x[:, None] - y) ** 2, dim=-1)
    return dist


def check_for_nan(tensor, label):
    if torch.isnan(tensor).any():
        print(f"{label} contains NaN values")


def iso_kernel(X, all_X, eta, psi):
    map_tmp = None
    if all_X is None:
        all_X = X
    samples_index = [
        np.random.choice(len(all_X), psi, replace=False) for _ in range(100)
    ]
    samples_index_set = np.concatenate(samples_index)
    unique_samples_index, indices = np.unique(samples_index_set, return_inverse=True)
    samples_index_inverse = np.split(indices, psi)
    scaling_factor = math.sqrt(X.shape[1])
    sim_unique_samples = torch.mm(all_X[unique_samples_index], X.T) / scaling_factor
    sim_samples = torch.tensor(
        [sim_unique_samples[index_inverse] for index_inverse in samples_index_inverse]
    )
    for s in sim_samples:
        log_soft_max_sim = torch.clamp(F.log_softmax(s, dim=1), min=-20, max=20) / 2
        soft_max_sim = torch.exp(log_soft_max_sim)
        if map_tmp is None:
            map_tmp = soft_max_sim
        else:
            map_tmp = torch.vstack([map_tmp, soft_max_sim])
    ik_similarity = torch.mm(map_tmp.T, map_tmp) / len(samples_index)
    assert ik_similarity.shape == (X.shape[0], X.shape[0])
    return ik_similarity
