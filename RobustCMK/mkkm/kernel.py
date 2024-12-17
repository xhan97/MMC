import torch
import numpy as np
import torch.nn.functional as F

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


def iso_kernel(X, all_X, eta, psi):
    map_tmp = None
    if all_X is None:
        all_X = X
    # samples_index = [torch.randperm(len(all_X))[:psi] for _ in range(100)]
    np.random.seed(42)
    samples_index = [
        np.random.choice(len(all_X), psi, replace=False) for _ in range(100)
    ]
    # print(samples_index)
    for s_index in samples_index:
        samples = all_X[s_index]
        dist = -2*eta*torch.cdist(X, samples)
        # dist -= torch.max(dist, dim=1)
        # soft_dist = torch.sqrt(torch.exp(-dist) / torch.sum(dist, dim=1, keepdim=True))
        soft_dist = torch.sqrt(torch.exp(F.log_softmax(dist, dim=1)))
        #(torch.exp(eta * dist) / torch.sqrt( torch.exp(2 * eta * dist).sum(dim=1) ).view(-1, 1))
        if map_tmp is None:
            map_tmp = soft_dist
        else:
            map_tmp = torch.hstack([map_tmp, soft_dist])
        if torch.mm(soft_dist, soft_dist.T).isinf().any():
            print(soft_dist)
        if torch.mm(soft_dist, soft_dist.T).isnan().any():
            print(soft_dist)
    return torch.mm(map_tmp, map_tmp.T) / len(samples_index)
