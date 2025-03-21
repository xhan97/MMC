import torch
import numpy as np
import torch.nn.functional as F
import math
from sklearn.datasets import load_iris
from typing import Optional, Tuple
import time


def check_for_nan(tensor: torch.Tensor, label: str) -> bool:
    """Check if tensor contains any NaN values and print a message if it does."""
    if torch.isnan(tensor).any():
        print(f"{label} contains NaN values")
        return True
    return False


def iso_kernel(
    X: torch.Tensor,
    all_X: Optional[torch.Tensor] = None,
    eta: float = 1.0,
    psi: int = 8,
    t: int = 100,
) -> torch.Tensor:
    """
    Implementation of the Isolation Kernel.

    Args:
        X: Input tensor of shape [n_samples, n_features]
        all_X: Reference tensor (defaults to X if None)
        eta: Scaling parameter for the softmax
        psi: Number of samples to select per iteration
        t: Number of iterations

    Returns:
        Kernel similarity matrix of shape [n_samples, n_samples]
    """
    if all_X is None:
        all_X = X

    device = X.device
    n_samples, n_features = X.shape
    scaling_factor = math.sqrt(n_features)

    # Generate all random indices at once
    samples_index = torch.tensor(
        np.array([np.random.choice(len(all_X), psi, replace=False) for _ in range(t)])
    ).to(device)

    # Use torch operations for better performance with GPU if available
    feature_maps = []

    # Process in smaller batches to avoid memory issues
    batch_size = min(t, 200)  # Adjust based on your memory constraints
    for i in range(0, t, batch_size):
        end_idx = min(i + batch_size, t)
        batch_indices = samples_index[i:end_idx]

        # Gather the selected samples for each batch
        batch_samples = all_X[batch_indices.flatten()].view(-1, psi, n_features)

        # Compute pairwise distances between batch_samples and X
        # Reshape for broadcasting: [batch, psi, features] and [samples, features]
        expanded_X = X.unsqueeze(0).unsqueeze(0)  # [1, 1, samples, features]
        expanded_samples = batch_samples.unsqueeze(2)  # [batch, psi, 1, features]

        # Calculate squared distances and normalize
        distances = (
            torch.sum((expanded_samples - expanded_X) ** 2, dim=3) / scaling_factor
        )
        similarities = -distances  # Convert distances to similarities

        # Apply softmax to get the feature map
        log_soft_max_sim = F.log_softmax(eta * similarities / 2, dim=1)
        log_soft_max_sim = torch.clamp(log_soft_max_sim, min=-20, max=20)
        soft_max_sim = torch.exp(log_soft_max_sim)

        if not check_for_nan(soft_max_sim, f"soft_max_sim in batch {i}"):
            feature_maps.append(soft_max_sim)

    if not feature_maps:
        raise ValueError("All feature maps contain NaN values")

    # Stack feature maps and compute kernel matrix
    map_tmp = torch.cat(feature_maps, dim=0)
    # Reshape to [t*psi, n_samples]
    map_tmp = map_tmp.view(-1, n_samples)
    ik_similarity = torch.mm(map_tmp.T, map_tmp) / t

    return ik_similarity


def iso_kernel_old(X, all_X=None, eta=1, psi=8, t=100):
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

    torch.manual_seed(42)
    np.random.seed(42)

    samples_index = np.array(
        [np.random.choice(len(all_X), psi, replace=False) for _ in range(t)]
    )
    unique_samples_index, indices = np.unique(
        samples_index.flatten(), return_inverse=True
    )
    samples_index_inverse = indices.reshape(t, psi)
    scaling_factor = math.sqrt(X.shape[1])
    sim_unique_samples = -(
        torch.cdist(all_X[unique_samples_index], X, p=2) / scaling_factor
    )
    feature_maps = []
    for idx in range(t):
        batch_indices = samples_index_inverse[idx]
        batch_sim = sim_unique_samples[batch_indices]
        log_soft_max_sim = torch.clamp(
            F.log_softmax(eta * batch_sim / 2, dim=0),
            min=-20,
            max=20,
        )
        soft_max_sim = torch.exp(log_soft_max_sim)
        if check_for_nan(soft_max_sim, f"soft_max_sim in iteration {idx}"):
            continue
        feature_maps.append(soft_max_sim)
    if not feature_maps:
        raise ValueError("All feature maps contain NaN values")
    map_tmp = torch.vstack(feature_maps)
    ik_similarity = torch.mm(map_tmp.T, map_tmp) / t
    assert ik_similarity.shape == (X.shape[0], X.shape[0]), "Invalid kernel shape"
    return ik_similarity


if __name__ == "__main__":
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Load data
    X, y = load_iris(return_X_y=True)
    X = torch.tensor(X, dtype=torch.float32).to(device)

    # Create tensor with gradient tracking
    Xs = X.clone().detach().requires_grad_(True)

    # Compute kernel and backpropagate
    st_time = time.time()
    x_map = iso_kernel(X=Xs, eta=100, psi=8, t=100)
    et_time = time.time()
    print(f"Time taken: {et_time - st_time:.6f} seconds")

    print(x_map)

    st_time = time.time()
    x_map_old = iso_kernel_old(X=Xs, eta=100, psi=8, t=100)
    ed_time = time.time()
    print(f"Time taken: {ed_time - st_time:.6f} seconds")
    print(x_map_old)

    # assert torch.equal(x_map, x_map_old), "Kernel mismatch"

    x_map_old.sum().backward()

    print("Gradient statistics:")
    print(f"Min: {Xs.grad.min().item():.6f}, Max: {Xs.grad.max().item():.6f}")
    print(f"Mean: {Xs.grad.mean().item():.6f}, Std: {Xs.grad.std().item():.6f}")
