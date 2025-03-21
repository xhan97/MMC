# Standard library imports
import os
import time
import math
import random
from argparse import ArgumentParser

import numpy as np
import scipy.io as scio
import torch
import torch.optim as optim
from sklearn.cluster import KMeans
from conloss import ConLoss
from model import FCNet
from metrics import MetricCalculator
from utils.files import load_data, save_results


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


def setup_environment(seed=42):
    """Set up the environment with reproducible random seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def initialize_model_components(args, feat_dims):
    """Initialize model, loss criterion and optimizer."""
    model = FCNet(
        feat_dims=feat_dims, latent_dim=args.latent_dim, normalize=args.normalize
    ).to(device=args.device)

    criterion = ConLoss(
        args.kernel_options, args.temperature, args.num_class, device=args.device
    ).to(device=args.device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    return model, criterion, optimizer


def perform_clustering(H, num_class, gt):
    """Perform clustering and evaluate performance."""
    kmeans = KMeans(n_clusters=num_class).fit(H.T)
    acc, nmi, pur = MetricCalculator.evaluate(gt, kmeans.labels_)
    return acc, nmi, pur


def update_metrics(
    metrics_history,
    optimizer,
    loss_con,
    loss_extra,
    k_diff,
    fea_diff,
    acc,
    nmi,
    pur,
    epoch_time,
):
    """Update the metrics history dictionary."""
    metrics_history["accs"].append(acc)
    metrics_history["nmis"].append(nmi)
    metrics_history["purs"].append(pur)
    metrics_history["lrs"].append(optimizer.param_groups[0]["lr"])
    metrics_history["losses_con"].append(loss_con)
    metrics_history["losses_extra"].append(loss_extra)
    metrics_history["k_diffs"].append(k_diff)
    metrics_history["fea_diffs"].append(fea_diff)
    metrics_history["time_epochs"].append(epoch_time)


# Training functions
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


def main(args):
    # Setup environment
    args.device = setup_environment()

    # Load data
    Xs, gt, num_class, feat_dims = load_data(args)
    args.num_class = num_class  # Store num_class in args for criterion initialization

    # Initialize model components
    model, criterion, optimizer = initialize_model_components(args, feat_dims)

    # Initialize metrics tracking
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

    # Training loop
    for epoch in range(1, args.epochs + 1):
        time1 = time.time()

        # Adjust learning rate and trade-off parameter
        adjust_learning_rate(args, optimizer, epoch)
        trade_off = 0 if epoch <= args.epochs // 3 else args.trade_off

        # Train for one epoch
        loss_con, loss_extra, kernels, k_diff, fea_diff, H, time_extra, time_extra_2 = (
            train(Xs, model, criterion, optimizer, trade_off)
        )

        # Calculate actual epoch time without extra calculations
        epoch_time = time.time() - time1 - time_extra - time_extra_2

        # Evaluate clustering performance
        acc, nmi, pur = perform_clustering(H, num_class, gt)

        # Update metrics history
        update_metrics(
            metrics_history,
            optimizer,
            loss_con,
            loss_extra,
            k_diff,
            fea_diff,
            acc,
            nmi,
            pur,
            epoch_time,
        )

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


# Entry point
if __name__ == "__main__":
    data_name = "BBCSport"
    results = main(default_args(data_name))
    print(f"Final results for {data_name}:")
    print(
        f"ACC: {results['acc']:.4f}, NMI: {results['nmi']:.4f}, PUR: {results['pur']:.4f}"
    )
