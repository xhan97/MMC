import time
import torch
import torch.nn as nn
import torch.nn.functional as F


from kernel import (
    gaussian_kernel,
    linear_kernel,
    polynomial_kernel,
    sigmoid_kernel,
    cauchy_kernel,
    iso_kernel,
)


class ConLoss(nn.Module):
    """Contrastive loss with kernel computation."""

    def __init__(
        self, kernel_options, temperature=1.0, num_class=10, device=torch.device("cpu")
    ):
        super(ConLoss, self).__init__()
        self.kernel_options = kernel_options
        self.temperature = temperature
        self.num_class = num_class
        self.device = device

    def forward(self, features):
        """Forward pass to compute contrastive loss with kernel computation."""
        num_view, num_smp = len(features), features[0].shape[0]
        features = torch.cat(features, dim=0)

        mask, logits_mask = self._create_mask(num_view, num_smp)
        K = self._compute_kernel(features)
        loss_con = self._calculate_contrastive_loss(K, mask, logits_mask)
        kernel = self._extract_view_kernels(K, num_smp, num_view)
        loss_extra, H = self._compute_clustering(kernel, num_smp)
        return loss_con, loss_extra, K.detach().cpu().numpy(), H, time.time()

    def _compute_kernel(self, features):
        """Compute kernel matrix based on kernel type."""
        kernel_type = self.kernel_options["type"]
        if kernel_type == "Gaussian":
            K = gaussian_kernel(
                torch.cat(features, dim=0),
                t=self.kernel_options["t"],
            )
        elif kernel_type == "Linear":
            K = linear_kernel(torch.cat(features, dim=0))
        elif kernel_type == "Polynomial":
            K = polynomial_kernel(
                torch.cat(features, dim=0),
                a=self.kernel_options["a"],
                b=self.kernel_options["b"],
                d=self.kernel_options["d"],
            )
        elif kernel_type == "Sigmoid":
            K = sigmoid_kernel(
                torch.cat(features, dim=0),
                d=self.kernel_options["d"],
                c=self.kernel_options["c"],
            )
        elif kernel_type == "Cauchy":
            K = cauchy_kernel(
                torch.cat(features, dim=0),
                sigma=self.kernel_options["sigma"],
            )
        elif kernel_type == "Isolation":
            K = iso_kernel(
                torch.cat(features, dim=0),
                eta=self.kernel_options["eta"],
                psi=self.kernel_options["psi"],
                t=self.kernel_options["t"],
            )
        else:
            raise NotImplementedError(f"Kernel type {kernel_type} not supported")
        return K

    def _create_mask(self, num_view, num_smp):
        """Create masks for positive pairs in contrastive loss."""
        mask = (
            torch.eye(num_smp, dtype=torch.float32)
            .to(self.device)
            .repeat(num_view, num_view)
        )
        logits_mask = torch.ones_like(mask).scatter_(
            1, torch.arange(num_smp * num_view).view(-1, 1).to(self.device), 0
        )
        mask *= logits_mask
        return mask, logits_mask

    def _calculate_contrastive_loss(self, K, mask, logits_mask):
        """Calculate the contrastive loss from kernel matrix and masks."""
        logits = torch.exp(K)
        log_prob = torch.log(logits) - torch.log(
            (logits * logits_mask).sum(1, keepdim=True)
        )
        mean_log_prob_pos = -(mask * log_prob).sum(1) / mask.sum(1)
        return mean_log_prob_pos.mean()

    def _extract_view_kernels(self, K, num_smp, num_view):
        """Extract and average kernel matrices across views."""
        kernels = torch.zeros([num_smp, num_smp, num_view], dtype=torch.float32).to(
            self.device
        )
        for i in range(num_view):
            kernels[:, :, i] = K[
                i * num_smp : (i + 1) * num_smp, i * num_smp : (i + 1) * num_smp
            ]
        return kernels.mean(2)

    def _compute_clustering(self, kernel, num_smp):
        """Compute eigenvalues/vectors for clustering and alignment loss."""
        val, vec = torch.linalg.eig(kernel.detach())
        val_real = val.real
        vec_real = vec.real
        _, ind = torch.sort(val_real, descending=True)
        H = vec_real[ind[: self.num_class]]

        loss_extra = (
            torch.trace(kernel) - torch.trace(torch.chain_matmul(H, kernel, H.T))
        ) / num_smp
        H = F.normalize(H).detach().cpu().numpy()

        return loss_extra, H
