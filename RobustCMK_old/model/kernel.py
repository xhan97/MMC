import torch
import torch.nn as nn
from mkkm.kernel import rbf_kernel, iso_kernel


class KernelLayer(nn.Module):
    def __init__(self, kernel_options: dict, device=torch.device("cpu")):
        super(KernelLayer, self).__init__()
        self.kernel_options = kernel_options
        self.device = device

    def forward(
        self,
        feature,
    ):
        features = torch.cat(feature, dim=0)

        # define Euclidean distance
        def EuDist2(fea_a, fea_b):
            return torch.cdist(fea_a, fea_b, p=2)

        # compute kernels
        if self.kernel_options["type"] == "rbf":
            K = rbf_kernel(features, gamma=self.kernel_options["gamma"])
        elif self.kernel_options["type"] == "Gaussian":
            D = EuDist2(features, features)
            K = torch.exp(-D / (2 * self.kernel_options["t"] ** 2))
        elif self.kernel_options["type"] == "Linear":
            K = torch.matmul(features, features.T)
        elif self.kernel_options["type"] == "Polynomial":
            K = torch.pow(
                self.kernel_options["a"] * torch.matmul(features, features.T)
                + self.kernel_options["b"],
                self.kernel_options["d"],
            )
        elif self.kernel_options["type"] == "Sigmoid":
            K = torch.tanh(
                self.kernel_options["d"] * torch.matmul(features, features.T)
                + self.kernel_options["c"]
            )
        elif self.kernel_options["type"] == "Cauchy":
            D = EuDist2(features, features)
            K = 1 / (D / self.kernel_options["sigma"] + 1)
        elif self.kernel_options["type"] == "ik":
            K = iso_kernel(
                features,
                features,
                self.kernel_options["eta"],
                self.kernel_options["sample_index"],
            )
        else:
            raise NotImplementedError
        return K
