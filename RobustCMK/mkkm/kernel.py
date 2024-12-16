import torch


# pytorch version
# Gaussian exp(-||x - y||^2/(2 * sigma^2))
def gaussian_kernel(x, y=None, sigma=None):
    if sigma is None:
        gamma = 1 / x.shape[1]
    else:
        gamma = 1 / (2 * sigma ** 2)
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
