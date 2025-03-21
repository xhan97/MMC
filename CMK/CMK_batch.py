import os
import time
import math
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn
import torch.nn.functional as F

import scipy.io as scio
import scipy.sparse as scsp
import h5py as hp
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.optimize import linear_sum_assignment


def read_mat(data_path, sparse=False):
    with hp.File(data_path, "r") as f:
        Y = f["Y"][()].T.astype(np.int32)
        Xr = f["X"][()].reshape((-1,)).tolist()
        X = []

        for x in Xr:
            if sparse:
                data = f[x]["data"][()]
                ir = f[x]["ir"][()]
                jc = f[x]["jc"][()]
                X.append(scsp.csc_matrix((data, ir, jc)).toarray())
            else:
                X.append(f[x][()].T.astype(np.float64))

    return X, Y


def load_data(args):
    data_path = os.path.join(args.data_dir, f"{args.data_name}.mat")
    try:
        data = scio.loadmat(data_path)
        Xs = data["X"].squeeze().tolist()
        gt = data["Y"].squeeze()
    except:
        try:
            Xs, gt = read_mat(data_path)
        except:
            Xs, gt = read_mat(data_path, sparse=True)

    num_class = np.unique(gt).shape[0]
    feat_dims = [x.shape[1] for x in Xs]

    Xs = [x.astype(np.float32).T for x in Xs]

    return Xs, gt, num_class, feat_dims


class AverageMeter:
    """Computes and stores the average and current values."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    return contingency_matrix[row_ind, col_ind].sum() / np.sum(contingency_matrix)


def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def nmi_score(y_true, y_pred):
    return metrics.normalized_mutual_info_score(y_true, y_pred)


def cluster_metric(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred)
    pur = purity_score(y_true, y_pred)
    return acc, nmi, pur


class FCNet(nn.Module):
    """Fully Connected Network with multiple views."""

    def __init__(self, feat_dims, latent_dim=64, normalize=True):
        super(FCNet, self).__init__()
        self.normalize = normalize
        self.layers = nn.ModuleList(
            [nn.Linear(dim, latent_dim, bias=False) for dim in feat_dims]
        )

    def forward(self, x):
        features = []
        weights = []
        for i, layer in enumerate(self.layers):
            feature = layer(x[i])
            if self.normalize:
                feature = F.normalize(feature, dim=1)
            features.append(feature)
            weights.append(layer.weight)
        return features, weights


class ConLoss(nn.Module):
    """Contrastive Loss for clustering."""

    def __init__(self, kernel_options, temperature=1.0, num_class=10, device="cpu"):
        super(ConLoss, self).__init__()
        self.kernel_options = kernel_options
        self.temperature = temperature
        self.num_class = num_class
        self.device = device

    def forward(self, features, trade_off):
        num_view, num_smp = len(features), features[0].shape[0]
        features = torch.cat(features, dim=0)

        mask = (
            torch.eye(num_smp, dtype=torch.float32)
            .to(self.device)
            .repeat(num_view, num_view)
        )
        logits_mask = torch.ones_like(mask).scatter(
            1, torch.arange(num_smp * num_view).view(-1, 1).to(self.device), 0
        )
        mask *= logits_mask

        D = self.euclidean_distance(features, features)
        K = self.compute_kernel(D)

        logits = torch.exp(K)
        log_prob = torch.log(logits) - torch.log(
            (logits * logits_mask).sum(1, keepdim=True)
        )
        mean_log_prob_pos = -(mask * log_prob).sum(1) / mask.sum(1)
        loss_con = mean_log_prob_pos.mean()

        loss_extra, H_out = self.compute_extra_loss(K, trade_off, num_smp)

        return (
            loss_con,
            loss_extra,
            K.detach().cpu().numpy(),
            H_out,
            0,
        )  # Time removed for brevity

    def euclidean_distance(self, fa, fb):
        aa = torch.sum(fa * fa, dim=1, keepdim=True)
        bb = torch.sum(fb * fb, dim=1, keepdim=True)
        ab = torch.matmul(fa, fb.T)
        return aa + bb.T - 2 * ab

    def compute_kernel(self, D):
        k_type = self.kernel_options["type"]
        if k_type == "Gaussian":
            return torch.exp(-D / (2 * self.kernel_options["t"] ** 2))
        elif k_type == "Linear":
            return torch.matmul(D, D.T)
        elif k_type == "Polynomial":
            return torch.pow(
                self.kernel_options["a"] * D + self.kernel_options["b"],
                self.kernel_options["d"],
            )
        elif k_type == "Sigmoid":
            return torch.tanh(self.kernel_options["d"] * D + self.kernel_options["c"])
        elif k_type == "Cauchy":
            return 1 / (D / self.kernel_options["sigma"] + 1)
        else:
            raise NotImplementedError

    def compute_extra_loss(self, K, trade_off, num_smp):
        if trade_off == 0:
            return torch.zeros(1, device=self.device), None

        kernels = K.view(len(K) // num_smp, num_smp, num_smp)
        kernel = kernels.mean(0)
        val, vec = np.linalg.eig(kernel.detach().cpu().numpy())
        ind = np.argsort(val.real)[::-1]
        H_out = vec.real[:, ind[: self.num_class]]
        H = torch.from_numpy(H_out).to(self.device)
        loss_extra = (
            torch.trace(kernel) - torch.trace(torch.chain_matmul(H.T, kernel, H))
        ) / num_smp
        H_out /= np.linalg.norm(H_out, axis=1, keepdims=True)
        return loss_extra, H_out


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate**3)
        lr = (
            eta_min
            + (lr - eta_min) * (1 + math.cos(math.pi * epoch * 3 / args.epochs)) / 2
        )
    else:
        steps = sum(epoch > np.array(args.lr_decay_epochs))
        if steps > 0:
            lr *= args.lr_decay_rate**steps

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr


def train_step(Xs, model, criterion, optimizer, trade_off, device):
    model.train()
    features, weights = model(Xs)
    loss_con, loss_extra, K, H, _ = criterion(features, trade_off)
    loss = loss_con + trade_off * loss_extra

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    feas = [fea.detach().cpu().numpy() for fea in features]
    kernels = K.reshape(len(features), Xs[0].shape[0], Xs[0].shape[0])

    k_diff = sum(
        np.power(kernels[i] - kernels[j], 2).mean()
        for i in range(len(features))
        for j in range(len(features))
        if i != j
    ) / (len(features) ** 2 - len(features))

    fea_diff = sum(
        np.power(feas[i] - feas[j], 2).mean()
        for i in range(len(features))
        for j in range(len(features))
        if i != j
    ) / (len(features) ** 2 - len(features))

    weights = [w.detach().cpu().numpy() for w in weights]
    return loss_con.item(), loss_extra.item(), k_diff, fea_diff, H, weights


def main(args):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    Xs, gt, num_class, feat_dims = load_data(args)

    model = FCNet(feat_dims, args.latent_dim, args.normalize).to(device)
    criterion = ConLoss(args.kernel_options, args.temperature, num_class, device).to(
        device
    )

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    for epoch in range(1, args.epochs + 1):
        lr = adjust_learning_rate(args, optimizer, epoch)
        trade_off = 0 if epoch <= args.epochs // 3 else args.trade_off

        rand_ind = np.random.permutation(Xs[0].shape[0])
        metrics = {
            "loss_con": AverageMeter(),
            "loss_extra": AverageMeter(),
            "k_diff": AverageMeter(),
            "fea_diff": AverageMeter(),
            "acc": AverageMeter(),
            "nmi": AverageMeter(),
            "pur": AverageMeter(),
        }

        for b in range(Xs[0].shape[0] // args.batch_size):
            batch_ind = rand_ind[b * args.batch_size : (b + 1) * args.batch_size]
            Xs_batch = [torch.tensor(x[batch_ind, :]).to(device) for x in Xs]

            loss_con, loss_extra, k_diff, fea_diff, H, weights = train_step(
                Xs_batch, model, criterion, optimizer, trade_off, device
            )

            metrics["loss_con"].update(loss_con, args.batch_size)
            metrics["loss_extra"].update(loss_extra, args.batch_size)
            metrics["k_diff"].update(k_diff, args.batch_size)
            metrics["fea_diff"].update(fea_diff, args.batch_size)

            if H is not None:
                kmeans = KMeans(n_clusters=num_class).fit(H.cpu().numpy())
                acc, nmi, pur = cluster_metric(gt[batch_ind], kmeans.labels_)
                metrics["acc"].update(acc, args.batch_size)
                metrics["nmi"].update(nmi, args.batch_size)
                metrics["pur"].update(pur, args.batch_size)

        if epoch % args.print_freq == 0:
            print(
                f"Epoch {epoch}: Loss Con={metrics['loss_con'].avg:.2f}, "
                f"Loss Extra={metrics['loss_extra'].avg:.2f}, "
                f"Acc={metrics['acc'].avg:.4f}"
            )

        # Save checkpoints if needed
        # ...

    torch.cuda.empty_cache()


def default_args(
    data_name, normalize, latent_dim, batch_size=2048, learning_rate=1.0, epochs=90
):
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])  # Empty args for default

    args.kernel_options = {"type": "Gaussian", "t": 1.0}
    args.normalize = normalize
    args.trade_off = 1
    args.latent_dim = latent_dim
    args.batch_size = batch_size
    args.learning_rate = learning_rate
    args.momentum = 0.9
    args.weight_decay = 0
    args.epochs = epochs
    assert args.epochs % 3 == 0
    args.cosine = True
    args.lr_decay_rate = 0.1
    args.lr_decay_epochs = [700, 800, 900]
    args.temperature = 1.0
    args.print_freq = 10
    args.data_dir = "./data"
    args.save_dir = "./save_batch_R1"
    args.data_name = data_name
    args.save_path = os.path.join(
        args.save_dir,
        args.data_name,
        f"norm_{args.normalize}",
        f"dim_{args.latent_dim}",
        f"batch_size_{args.batch_size}",
        f"lr_{args.learning_rate}",
        f"epochs_{args.epochs}",
    )
    os.makedirs(args.save_path, exist_ok=True)

    return args


if __name__ == "__main__":
    data_name = "bbcsport_2view"
    args = default_args(data_name, normalize=True, latent_dim=64)
    main(args)
