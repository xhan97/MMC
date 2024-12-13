import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import scipy.io as scio
from sklearn.cluster import KMeans
import sklearn.metrics as metrics
import argparse
from scipy.optimize import linear_sum_assignment


def load_data(args):
    data = scio.loadmat(os.path.join(args.data_dir, args.data_name + ".mat"))
    Xs = data["X"].squeeze().tolist()
    gt = data["y"].squeeze()
    num_class = np.unique(gt).shape[0]
    feat_dims = [
        torch.tensor(X.astype(np.float32).T, dtype=torch.float32)
        .to(args.device)
        .shape[1]
        for X in Xs
    ]
    Xs = [
        torch.tensor(X.astype(np.float32).T, dtype=torch.float32).to(args.device)
        for X in Xs
    ]
    return Xs, gt, num_class, feat_dims


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
    def __init__(self, feat_dims=[100,100], latent_dim=64, normalize=True):
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


def IK_Kernel(X, eta, samples_index):
    map_tmp = None
    for s_index in samples_index:
        samples = X[s_index]
        dist = torch.cdist(X, samples)
        soft_dist = torch.exp(-eta * dist) / torch.sqrt(
            torch.exp(-2 * eta * dist).sum(dim=1)
        ).view(-1, 1)
        if map_tmp is None:
            map_tmp = soft_dist
        else:
            map_tmp = torch.hstack([map_tmp, soft_dist])
    return torch.mm(map_tmp, map_tmp.T) / len(samples_index)


class ConLoss(nn.Module):
    def __init__(
        self, kernel_options, temperature=1.0, num_class=10, device=torch.device("cpu")
    ):
        super(ConLoss, self).__init__()
        self.kernel_options = kernel_options
        self.temperature = temperature
        self.num_class = num_class
        self.device = device

    def forward(self, features):
        num_view, num_smp = len(features), features[0].shape[0]
        features = torch.cat(features, dim=0)
        mask = (
            torch.eye(num_smp, dtype=torch.float32)
            .to(self.device)
            .repeat(num_view, num_view)
        )
        logits_mask = torch.ones_like(mask).scatter_(
            1, torch.arange(num_smp * num_view).view(-1, 1).to(self.device), 0
        )
        mask *= logits_mask

        def EuDist2(fea_a, fea_b):
            aa = torch.sum(fea_a * fea_a, 1, keepdim=True)
            bb = torch.sum(fea_b * fea_b, 1, keepdim=True)
            ab = torch.matmul(fea_a, fea_b.T)
            return aa + bb.T - 2 * ab

        if self.kernel_options["type"] == "Gaussian":
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
        elif self.kernel_options["type"] == "IK":
            K = IK_Kernel(
                features,
                self.kernel_options["eta"],
                self.kernel_options["samples_index"],
            )
        else:
            raise NotImplementedError

        logits = torch.exp(K)
        log_prob = torch.log(logits) - torch.log(
            (logits * logits_mask).sum(1, keepdim=True)
        )
        mean_log_prob_pos = -(mask * log_prob).sum(1) / mask.sum(1)
        loss_con = mean_log_prob_pos.mean()

        kernels = torch.zeros([num_smp, num_smp, num_view], dtype=torch.float32).to(
            self.device
        )
        for i in range(num_view):
            kernels[:, :, i] = K[
                i * num_smp : (i + 1) * num_smp, i * num_smp : (i + 1) * num_smp
            ]
        kernel = kernels.mean(2)
        val, vec = torch.eig(kernel.detach(), eigenvectors=True)
        _, ind = torch.sort(val[:, 0], descending=True)
        H = vec[:, ind[: self.num_class]]

        loss_extra = (
            torch.trace(kernel) - torch.trace(torch.chain_matmul(H.T, kernel, H))
        ) / num_smp
        H = F.normalize(H).detach().cpu().numpy()

        return loss_con, loss_extra, K.detach().cpu().numpy(), H, time.time()


def adjust_learning_rate(args, optimizer, epoch):
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
    model.train()
    features = model(Xs)
    loss_con, loss_extra, K, H, time_extra = criterion(features)
    loss = loss_con + trade_off * loss_extra

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    num_smp, num_view = Xs[0].shape[0], len(Xs)
    feas = [fea.detach().cpu().numpy() for fea in features]

    kernels = np.zeros([num_smp, num_smp, num_view])
    for i in range(num_view):
        kernels[:, :, i] = K[
            i * num_smp : (i + 1) * num_smp, i * num_smp : (i + 1) * num_smp
        ]

    k_diff, fea_diff = 0, 0
    for i in range(num_view):
        for j in range(num_view):
            if i != j:
                k_diff += np.power(kernels[:, :, i] - kernels[:, :, j], 2).mean() / (
                    num_view**2 - num_view
                )
                fea_diff += np.power(feas[i] - feas[j], 2).mean() / (
                    num_view**2 - num_view
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
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Xs, gt, num_class, feat_dims = load_data(args)

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

    time_running = time.time()
    lrs, losses_con, losses_extra, k_diffs, fea_diffs, accs, nmis, purs = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    time_epochs = []

    for epoch in range(1, args.epochs + 1):
        time1 = time.time()
        adjust_learning_rate(args, optimizer, epoch)
        trade_off = 0 if epoch <= args.epochs // 3 else args.trade_off

        loss_con, loss_extra, kernels, k_diff, fea_diff, H, time_extra, time_extra_2 = (
            train(Xs, model, criterion, optimizer, trade_off)
        )
        time_epochs.append(time.time() - time1 - time_extra - time_extra_2)

        kmeans = KMeans(n_clusters=num_class).fit(H)
        acc, nmi, pur = cluster_metric(gt, kmeans.labels_)
        accs.append(acc)
        nmis.append(nmi)
        purs.append(pur)

        lrs.append(optimizer.param_groups[0]["lr"])
        losses_con.append(loss_con)
        losses_extra.append(loss_extra)
        k_diffs.append(k_diff)
        fea_diffs.append(fea_diff)

        if epoch % args.print_freq == 0:
            print(
                f"  . epoch {epoch}, time: {time.time() - time_running:.2f}, loss_con: {loss_con:.2f}, loss_extra: {loss_extra:.2f}"
            )

        save_file = None
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

        save_file_2 = None
        if epoch in args.save_epochs:
            save_file_2 = os.path.join(
                args.save_path, f"{args.kernel_options['type']}_epoch_{epoch}.mat"
            )

        save_dict = {
            "accs": accs,
            "nmis": nmis,
            "purs": purs,
            "gt": gt,
            "lrs": lrs,
            "losses_con": losses_con,
            "losses_extra": losses_extra,
            "k_diffs": k_diffs,
            "fea_diffs": fea_diffs,
            "time_epochs": time_epochs,
            "K": kernels,
        }

        if save_file:
            scio.savemat(save_file, save_dict)
        if save_file_2:
            scio.savemat(save_file_2, save_dict)

    torch.cuda.empty_cache()


def default_args(
    data_name,
    normalize=True,
    latent_dim=128,
    learning_rate=1.0,
    epochs=300,
    save_epochs=[],
):
    args = argparse.ArgumentParser().parse_args()
    args.kernel_options = {"type": "Gaussian", "t": 1.0}
    args.normalize = normalize
    args.trade_off = 1
    args.latent_dim = latent_dim
    args.learning_rate = learning_rate
    args.momentum = 0.9
    args.weight_decay = 0
    args.epochs = epochs
    assert args.epochs % 3 == 0
    args.cosine = True
    args.lr_decay_rate = 0.1
    args.lr_decay_epochs = [700, 800, 900]
    args.temperature = 1.0
    args.print_freq = 100
    args.save_epochs = save_epochs
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
    main(default_args(data_name))
