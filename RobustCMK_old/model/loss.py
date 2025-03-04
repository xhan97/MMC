import torch
import torch.nn as nn
from mkkm.kernel import rbf_kernel, iso_kernel
import torch.nn.functional as F


# copy from "Contrastive Multi-view Kernel Learning"
class CMKLoss(nn.Module):
    def __init__(self, kernel_options, device=torch.device("cpu")):
        super(CMKLoss, self).__init__()
        self.kernel_options = kernel_options
        self.device = device

    def forward(
        self, feature, pos_mask, neg_mask, true_neg_mask, false_neg_mask, **kwargs
    ):
        # flatten features
        num_view, num_smp = len(feature), feature[0].shape[0]
        features = torch.cat(feature, dim=0)

        mask = pos_mask
        logits_mask = pos_mask + neg_mask

        # define Euclidean distance
        def EuDist2(fea_a, fea_b):
            num_smp = fea_a.shape[0]
            aa = torch.sum(fea_a * fea_a, 1, keepdim=True)
            bb = torch.sum(fea_b * fea_b, 1, keepdim=True)
            ab = torch.matmul(fea_a, fea_b.T)
            D = aa.repeat([1, num_smp]) + bb.repeat([1, num_smp]) - 2 * ab
            return D

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
        else:
            raise NotImplementedError
        # loss of contrastive learning
        logits = torch.exp(K)
        log_prob = torch.log(logits) - torch.log(
            (logits * logits_mask).sum(1, keepdim=True)
        )
        mean_log_prob_pos = -(mask * log_prob).sum(1) / mask.sum(1)
        loss_con = mean_log_prob_pos.mean()

        with torch.no_grad():
            pos_avg = torch.sum(K * pos_mask) / torch.sum(pos_mask)
            neg_avg = torch.sum(K * neg_mask) / torch.sum(neg_mask)
            true_neg_avg = torch.sum(K * true_neg_mask) / torch.sum(true_neg_mask)
            false_neg_avg = torch.sum(K * false_neg_mask) / torch.sum(false_neg_mask)
        return (
            K.detach(),
            loss_con,
            pos_avg.item(),
            neg_avg.item(),
            true_neg_avg.item(),
            false_neg_avg.item(),
        )


class RCMKLoss(nn.Module):
    def __init__(
        self, kernel_options: dict, num_class, trade_off, device=torch.device("cpu")
    ):
        super(RCMKLoss, self).__init__()
        self.kernel_options = kernel_options
        self.device = device
        self.num_class = num_class
        self.trade_off = trade_off

    def forward(
        self, features, pos_mask, neg_mask, true_neg_mask, false_neg_mask, **kwargs
    ):
        m = kwargs["m"]
        num_view, num_smp = len(features), features[0].shape[0]
        K = self.compute_kernel_matrix(features)
        dist = torch.exp(-K)
        loss_con = self.calculate_contrastive_loss(pos_mask, neg_mask, m, dist)
        H, loss_extra = self.calculate_extra_loss(num_view, num_smp, K)
        loss_all = loss_con + self.trade_off * loss_extra
        with torch.no_grad():
            pos_avg = torch.sum(dist * pos_mask) / torch.sum(pos_mask)
            neg_avg = torch.sum(dist * neg_mask) / torch.sum(neg_mask)
            true_neg_avg = torch.sum(dist * true_neg_mask) / torch.sum(true_neg_mask)
            false_neg_avg = torch.sum(dist * false_neg_mask) / torch.sum(false_neg_mask)
        return (
            H,
            loss_all,
            pos_avg.item(),
            neg_avg.item(),
            true_neg_avg.item(),
            false_neg_avg.item(),
        )

    def calculate_extra_loss(self, num_view, num_smp, K):
        K_patches = K.unfold(0, num_smp, num_smp).unfold(1, num_smp, num_smp)
        kernel = torch.stack([K_patches[i, i] for i in range(num_view)], dim=0).mean(0)
        val, vec = torch.linalg.eig(kernel.detach())
        _, ind = torch.sort(torch.real(val.detach().cpu()), descending=True)
        H = torch.real(vec[:, ind[: self.num_class]])
        loss_extra = (
            torch.trace(kernel) - torch.trace(torch.chain_matmul(H, kernel, H.T))
        ) / num_smp
        H = F.normalize(H).detach().cpu().numpy()
        return H, loss_extra

    def calculate_contrastive_loss(self, pos_mask, neg_mask, m, dist):
        pos_loss = dist**2
        neg_loss = (1 / m) * torch.pow(
            torch.clamp(torch.pow(dist, 0.5) * (m - dist), min=0.0), 2
        )
        loss_con = torch.sum(pos_loss * pos_mask + neg_loss * neg_mask) / torch.sum(
            pos_mask + neg_mask
        )
        return loss_con

    def EuDist2(self, fea_a: torch.Tensor, fea_b: torch.Tensor) -> torch.Tensor:
        return torch.cdist(fea_a.to(self.device), fea_b.to(self.device), p=2)

    def compute_kernel_matrix(self, features: torch.Tensor) -> torch.Tensor:
        features = features.to(self.device)
        kernel_type = self.kernel_options["type"].lower()

        if kernel_type == "rbf":
            return rbf_kernel(features, gamma=self.kernel_options["gamma"])
        elif kernel_type == "gaussian":
            D = self.EuDist2(features, features)
            return torch.exp(-D / (2 * self.kernel_options["t"] ** 2))
        elif kernel_type == "linear":
            return torch.matmul(features, features.T)
        elif kernel_type == "polynomial":
            return torch.pow(
                self.kernel_options["a"] * torch.matmul(features, features.T)
                + self.kernel_options["b"],
                self.kernel_options["d"],
            )
        elif kernel_type == "sigmoid":
            return torch.tanh(
                self.kernel_options["d"] * torch.matmul(features, features.T)
                + self.kernel_options["c"]
            )
        elif kernel_type == "cauchy":
            D = self.EuDist2(features, features)
            return 1 / (D / self.kernel_options["sigma"] + 1)
        elif kernel_type == "ik":
            return iso_kernel(
                features,
                all_X=None,
                eta=self.kernel_options["eta"],
                psi=self.kernel_options["psi"],
            )
        else:
            raise NotImplementedError(
                f"Kernel type '{kernel_type}' is not implemented."
            )
