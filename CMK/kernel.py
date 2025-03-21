import torch
import numpy as np
import torch.nn.functional as F
import math
from typing import Optional


def gaussian_kernel(
    X: torch.Tensor, Y: torch.Tensor, sigma: float = 1.0
) -> torch.Tensor:
    """
    Implementation of the Gaussian kernel.

    Args:
        X: Input tensor of shape [n_samples, n_features]
        Y: Input tensor of shape [m_samples, n_features]
        sigma: Kernel bandwidth

    Returns:
        Kernel similarity matrix of shape [n_samples, m_samples]
    """
    X_norm = torch.sum(X**2, dim=1, keepdim=True)
    Y_norm = torch.sum(Y**2, dim=1, keepdim=True).T
    dist = X_norm + Y_norm - 2 * torch.matmul(X, Y.T)
    # Clamp small negative values to zero for numerical stability
    dist = torch.clamp(dist, min=0.0)
    return torch.exp(-dist / (2 * sigma**2))


def linear_kernel(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Implementation of the Linear kernel.

    Args:
        X: Input tensor of shape [n_samples, n_features]
        Y: Input tensor of shape [m_samples, n_features]

    Returns:
        Kernel similarity matrix of shape [n_samples, m_samples]
    """
    return torch.matmul(X, Y.T)


def polynomial_kernel(
    X: torch.Tensor, Y: torch.Tensor, a: float = 1.0, b: float = 1.0, d: int = 2
) -> torch.Tensor:
    """
    Implementation of the Polynomial kernel.

    Args:
        X: Input tensor of shape [n_samples, n_features]
        Y: Input tensor of shape [m_samples, n_features]
        a: Scaling parameter
        b: Bias parameter
        d: Degree parameter

    Returns:
        Kernel similarity matrix of shape [n_samples, m_samples]
    """
    return (a * torch.matmul(X, Y.T) + b) ** d


def sigmoid_kernel(
    X: torch.Tensor, Y: torch.Tensor, c: float = 1.0, d: float = 0.0
) -> torch.Tensor:
    """
    Implementation of the Sigmoid kernel.

    Args:
        X: Input tensor of shape [n_samples, n_features]
        Y: Input tensor of shape [m_samples, n_features]
        c: Scaling parameter
        d: Bias parameter

    Returns:
        Kernel similarity matrix of shape [n_samples, m_samples]
    """
    return torch.tanh(c * torch.matmul(X, Y.T) + d)


def cauchy_kernel(X: torch.Tensor, Y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Implementation of the Cauchy kernel.

    Args:
        X: Input tensor of shape [n_samples, n_features]
        Y: Input tensor of shape [m_samples, n_features]
        sigma: Kernel bandwidth

    Returns:
        Kernel similarity matrix of shape [n_samples, m_samples]
    """
    # More efficient pairwise distance calculation
    dist_squared = torch.cdist(X, Y, p=2).pow(2)
    return 1 / (1 + dist_squared / sigma**2)


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
        eta: Scaling parameter
        psi: Number of samples to select per iteration
        t: Number of iterations

    Returns:
        Kernel similarity matrix of shape [n_samples, n_samples]
    """
    if all_X is None:
        all_X = X

    device = X.device

    # Use torch's random number generator for better reproducibility on GPU
    generator = torch.Generator(device=device if device.type != "mps" else "cpu")
    generator.manual_seed(42)

    n_samples = len(all_X)
    scaling_factor = math.sqrt(X.shape[1])

    feature_maps = []
    for _ in range(t):
        # Sample indices using torch's random generator
        if device.type == "cuda":
            samples_indices = torch.randperm(
                n_samples, generator=generator, device=device
            )[:psi]
        else:
            samples_indices = torch.randperm(n_samples, generator=generator)[:psi].to(
                device
            )

        # Compute distances using selected samples
        batch_samples = all_X[samples_indices]
        batch_sim = -(torch.cdist(batch_samples, X, p=2) / scaling_factor)

        # Apply softmax and convert to feature map
        log_soft_max_sim = F.log_softmax(eta * batch_sim / 2, dim=0).clamp(
            min=-20, max=20
        )
        soft_max_sim = torch.exp(log_soft_max_sim)
        feature_maps.append(soft_max_sim)

    if not feature_maps:
        raise ValueError("No feature maps were generated")

    # Stack feature maps and compute kernel matrix
    map_tmp = torch.cat(feature_maps, dim=0)
    ik_similarity = torch.matmul(map_tmp.T, map_tmp) / t

    return ik_similarity
