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


def iso_kernel(X, all_X=None, eta=1, psi=8, t=100):
    """
    Implementation of the Isolation Kernel.

    Args:
        X: Input tensor of shape [n_samples, n_features]
        all_X: Reference tensor (defaults to X if None)
        eta: Scaling parameter
        psi: Number of samples to select per iteration
        t: Number of iterations

    Returns:
        Kernel similarity matrix of shape [n_samples, n_samples]
    """
    if all_X is None:
        all_X = X

    np.random.seed(42)
    samples_index = np.array(
        [np.random.choice(len(all_X), psi, replace=False) for _ in range(t)]
    )
    unique_samples_index, indices = np.unique(
        samples_index.flatten(), return_inverse=True
    )
    samples_index_inverse = indices.reshape(t, psi)
    scaling_factor = math.sqrt(X.shape[1])
    # Compute similarities between unique samples and X
    # sim_unique_samples = torch.mm(all_X[unique_samples_index], X.T) / scaling_factor
    sim_unique_samples = -(
        torch.cdist(all_X[unique_samples_index], X, p=2) / scaling_factor
    )
    # Process each batch of samples
    feature_maps = []
    for idx in range(t):
        batch_indices = samples_index_inverse[idx]
        batch_sim = sim_unique_samples[batch_indices]

        # Apply softmax and take square root for the feature map
        log_soft_max_sim = torch.clamp(
            F.log_softmax(eta * batch_sim / 2, dim=0), min=-20, max=20
        )
        soft_max_sim = torch.exp(log_soft_max_sim)

        if check_for_nan(soft_max_sim, f"soft_max_sim in iteration {idx}"):
            continue

        feature_maps.append(soft_max_sim)

    if not feature_maps:
        raise ValueError("All feature maps contain NaN values")

    # Stack feature maps and compute kernel matrix
    map_tmp = torch.vstack(feature_maps)
    ik_similarity = torch.matmul(map_tmp.T, map_tmp) / t

    assert ik_similarity.shape == (X.shape[0], X.shape[0]), "Invalid kernel shape"
    return ik_similarity
