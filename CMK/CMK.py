# Reorganize imports by type
import os
import time
import math
import random
from argparse import ArgumentParser

# Third-party imports
import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
import sklearn.metrics as metrics
from scipy.optimize import linear_sum_assignment
from kernel import (
    gaussian_kernel,
    linear_kernel,
    polynomial_kernel,
    sigmoid_kernel,
    cauchy_kernel,
    iso_kernel,
)


def load_data(args):
    """Load multi-view data from .mat file."""
    data = scio.loadmat(os.path.join(args.data_dir, args.data_name + ".mat"))
    Xs = data["X"].squeeze().tolist()
    gt = data["y"].squeeze()
    num_class = np.unique(gt).shape[0]

    # Convert to torch tensors
    feat_dims = [
        torch.tensor(X.astype(np.float32)).to(args.device).shape[1] for X in Xs
    ]
    Xs = [torch.tensor(X.astype(np.float32)).to(args.device) for X in Xs]

    return Xs, gt, num_class, feat_dims


class MetricCalculator:
    """Class for computing clustering metrics."""

    @staticmethod
    def accuracy_score(y_true, y_pred):
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
        return contingency_matrix[row_ind, col_ind].sum() / np.sum(contingency_matrix)

    @staticmethod
    def purity_score(y_true, y_pred):
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

    @staticmethod
    def nmi_score(y_true, y_pred):
        return metrics.normalized_mutual_info_score(y_true, y_pred)

    @classmethod
    def evaluate(cls, y_true, y_pred):
        """Compute all metrics at once."""
        acc = cls.accuracy_score(y_true, y_pred)
        nmi = cls.nmi_score(y_true, y_pred)
        pur = cls.purity_score(y_true, y_pred)
        return acc, nmi, pur


class FCNet(nn.Module):
    """Fully connected network for feature extraction."""

    def __init__(self, feat_dims, latent_dim=64, normalize=True):
        super(FCNet, self).__init__()
        self.feat_dims = feat_dims
        self.num_view = len(feat_dims)
        self.latent_dim = latent_dim
        self.normalize = normalize
        self.fc_layers = nn.ModuleList(
            [nn.Linear(feat_dim, latent_dim, bias=False) for feat_dim in feat_dims]
        )

    def forward(self, x):
        return [
            F.normalize(fc(x_i), dim=1) if self.normalize else fc(x_i)
            for fc, x_i in zip(self.fc_layers, x)
        ]


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

    def forward(self, features):
        num_view, num_smp = len(features), features[0].shape[0]
        features = torch.cat(features, dim=0)

        # Create mask for positive pairs
        mask = (
            torch.eye(num_smp, dtype=torch.float32)
            .to(self.device)
            .repeat(num_view, num_view)
        )
        logits_mask = torch.ones_like(mask).scatter_(
            1, torch.arange(num_smp * num_view).view(-1, 1).to(self.device), 0
        )
        mask *= logits_mask

        # Compute kernel matrix
        K = self._compute_kernel(features)

        # Compute contrastive loss
        logits = torch.exp(K)
        log_prob = torch.log(logits) - torch.log(
            (logits * logits_mask).sum(1, keepdim=True)
        )
        mean_log_prob_pos = -(mask * log_prob).sum(1) / mask.sum(1)
        loss_con = mean_log_prob_pos.mean()

        # Extract kernels per view
        kernels = torch.zeros([num_smp, num_smp, num_view], dtype=torch.float32).to(
            self.device
        )
        for i in range(num_view):
            kernels[:, :, i] = K[
                i * num_smp : (i + 1) * num_smp, i * num_smp : (i + 1) * num_smp
            ]
        kernel = kernels.mean(2)

        # Compute eigenvalues/vectors for clustering
        val, vec = torch.linalg.eig(kernel.detach())
        val_real = val.real
        vec_real = vec.real
        _, ind = torch.sort(val_real, descending=True)
        H = vec_real[ind[: self.num_class]]

        # Extra loss for kernel alignment
        loss_extra = (
            torch.trace(kernel) - torch.trace(torch.chain_matmul(H, kernel, H.T))
        ) / num_smp
        H = F.normalize(H).detach().cpu().numpy()

        return loss_con, loss_extra, K.detach().cpu().numpy(), H, time.time()


def adjust_learning_rate(args, optimizer, epoch):
    """Adjust learning rate during training."""
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate**3)
        lr = (
            eta_min
            + (lr - eta_min) * (1 + math.cos(math.pi * epoch * 3 / args.epochs)) / 2
        )
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr *= args.lr_decay_rate**steps

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train(Xs, model, criterion, optimizer, trade_off):
    """Train model for one epoch."""
    model.train()

    # Forward pass
    features = model(Xs)
    loss_con, loss_extra, K, H, time_extra = criterion(features)
    loss = loss_con + trade_off * loss_extra

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Calculate difference metrics
    num_smp, num_view = Xs[0].shape[0], len(Xs)
    feas = [fea.detach().cpu().numpy() for fea in features]

    # Extract kernels per view
    kernels = np.zeros([num_smp, num_smp, num_view])
    for i in range(num_view):
        kernels[:, :, i] = K[
            i * num_smp : (i + 1) * num_smp, i * num_smp : (i + 1) * num_smp
        ]

    # Compute differences between views
    k_diff, fea_diff = 0, 0
    for i in range(num_view):
        for j in range(i + 1, num_view):  # Optimize: only compute unique pairs
            k_diff += (
                np.power(kernels[:, :, i] - kernels[:, :, j], 2).mean()
                * 2
                / (num_view**2 - num_view)
            )
            fea_diff += (
                np.power(feas[i] - feas[j], 2).mean() * 2 / (num_view**2 - num_view)
            )

    return (
        loss_con.item(),
        loss_extra.item(),
        kernels,
        k_diff,
        fea_diff,
        H,
        time_extra,
        time.time(),
    )


def save_results(epoch, args, save_dict):
    """Save results to files based on epoch."""
    save_file = None

    # Regular checkpoints
    if epoch == args.epochs // 3:
        save_file = os.path.join(
            args.save_path, f"{args.kernel_options['type']}_cmk.mat"
        )
    elif epoch == args.epochs * 2 // 3:
        save_file = os.path.join(
            args.save_path, f"{args.kernel_options['type']}_cmkkm_mid.mat"
        )
    elif epoch == args.epochs:
        save_file = os.path.join(
            args.save_path, f"{args.kernel_options['type']}_cmkkm.mat"
        )

    # Custom checkpoints
    save_file_2 = None
    if epoch in args.save_epochs:
        save_file_2 = os.path.join(
            args.save_path, f"{args.kernel_options['type']}_epoch_{epoch}.mat"
        )

    if save_file:
        scio.savemat(save_file, save_dict)
    if save_file_2:
        scio.savemat(save_file_2, save_dict)


def main(args):
    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # Set device
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load data
    Xs, gt, num_class, feat_dims = load_data(args)

    # Initialize model, criterion and optimizer
    model = FCNet(
        feat_dims=feat_dims, latent_dim=args.latent_dim, normalize=args.normalize
    ).to(device=args.device)

    criterion = ConLoss(
        args.kernel_options, args.temperature, num_class, device=args.device
    ).to(device=args.device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Training loop
    time_running = time.time()
    metrics_history = {
        "lrs": [],
        "losses_con": [],
        "losses_extra": [],
        "k_diffs": [],
        "fea_diffs": [],
        "accs": [],
        "nmis": [],
        "purs": [],
        "time_epochs": [],
    }

    for epoch in range(1, args.epochs + 1):
        time1 = time.time()

        # Adjust learning rate and trade-off parameter
        adjust_learning_rate(args, optimizer, epoch)
        trade_off = 0 if epoch <= args.epochs // 3 else args.trade_off

        # Train for one epoch
        loss_con, loss_extra, kernels, k_diff, fea_diff, H, time_extra, time_extra_2 = (
            train(Xs, model, criterion, optimizer, trade_off)
        )
        metrics_history["time_epochs"].append(
            time.time() - time1 - time_extra - time_extra_2
        )

        # Evaluate clustering performance
        kmeans = KMeans(n_clusters=num_class).fit(H.T)
        acc, nmi, pur = MetricCalculator.evaluate(gt, kmeans.labels_)

        # Record metrics
        metrics_history["accs"].append(acc)
        metrics_history["nmis"].append(nmi)
        metrics_history["purs"].append(pur)
        metrics_history["lrs"].append(optimizer.param_groups[0]["lr"])
        metrics_history["losses_con"].append(loss_con)
        metrics_history["losses_extra"].append(loss_extra)
        metrics_history["k_diffs"].append(k_diff)
        metrics_history["fea_diffs"].append(fea_diff)

        # Print progress
        if epoch % args.print_freq == 0:
            print(
                f"Epoch {epoch:3d}/{args.epochs}, "
                f"Time: {time.time() - time_running:.2f}s, "
                f"Loss_con: {loss_con:.4f}, Loss_extra: {loss_extra:.4f}, "
                f"ACC: {acc:.4f}, NMI: {nmi:.4f}, PUR: {pur:.4f}"
            )

        # Save results
        save_dict = {
            **metrics_history,
            "gt": gt,
            "K": kernels,
        }
        save_results(epoch, args, save_dict)

    torch.cuda.empty_cache()

    # Return final results
    return {
        "acc": metrics_history["accs"][-1],
        "nmi": metrics_history["nmis"][-1],
        "pur": metrics_history["purs"][-1],
    }


def default_args(
    data_name,
    normalize=True,
    latent_dim=128,
    learning_rate=1.0,
    epochs=300,
    save_epochs=None,
):
    """Create default arguments for the model."""
    parser = ArgumentParser(description="Contrastive Multi-view Kernel Framework")
    args = parser.parse_args([])

    # Model parameters
    args.kernel_options = {"type": "Gaussian", "t": 1.0}
    args.normalize = normalize
    args.trade_off = 1
    args.latent_dim = latent_dim
    args.temperature = 1.0

    # Training parameters
    args.learning_rate = learning_rate
    args.momentum = 0.9
    args.weight_decay = 0
    args.epochs = epochs
    if args.epochs % 3 != 0:
        raise ValueError("Epochs must be divisible by 3")
    args.cosine = True
    args.lr_decay_rate = 0.1
    args.lr_decay_epochs = [700, 800, 900]
    args.print_freq = 100
    args.save_epochs = save_epochs or []

    # Paths
    args.data_dir = "datasets"
    args.save_dir = "save"
    args.data_name = data_name
    args.save_path = os.path.join(
        args.save_dir,
        args.data_name,
        f"norm_{args.normalize}",
        f"dim_{args.latent_dim}",
        f"lr_{args.learning_rate}",
        f"epochs_{args.epochs}",
    )
    os.makedirs(args.save_path, exist_ok=True)

    return args


if __name__ == "__main__":
    data_name = "BBCSport"
    results = main(default_args(data_name))
    print(f"Final results for {data_name}:")
    print(
        f"ACC: {results['acc']:.4f}, NMI: {results['nmi']:.4f}, PUR: {results['pur']:.4f}"
    )
