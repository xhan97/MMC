import os
import scipy.io as scio
import numpy as np
import torch


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
