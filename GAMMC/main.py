import torch
import os
from sklearn.cluster import KMeans
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import opt
from utils import calculate_metrics, check_dir_exist
from model import IDMMC
from utils import MFeatDataSet, SFeatDataSet
from utils import calculate_metrics, check_dir_exist
from torch.utils.data import DataLoader
import time
import numpy as np

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

METRIC_PRINT = "metrics: " + ", ".join(["{:.4f}"] * 7)


def off_diagonal(x):
    """
    off-diagonal elements of x
    Args:
        x: the input matrix
    Returns: the off-diagonal elements of x
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def train_IDMMC(model, feats, modalitys, labels):

    optimizer = optim.Adam(model.parameters(), lr=opt.args.lr_m)

    for epoch in range(opt.args.n_epochs):
        h_img_all, h_txt_all, h_fuse, h2_img_feat, h2_txt_feat = model(
            feats, modalitys, opt.args.k1, opt.args.k2, opt.args.k3
        )

        img_recon_loss = F.mse_loss(h_img_all[0], h_img_all[1])
        txt_recon_loss = F.mse_loss(h_txt_all[0], h_txt_all[1])
        img_cycle_loss = F.l1_loss(h_img_all[2], h_img_all[3])
        txt_cycle_loss = F.l1_loss(h_txt_all[2], h_txt_all[3])
        img_cycle_recon_loss = F.mse_loss(h_img_all[0], h_img_all[4])
        txt_cycle_recon_loss = F.mse_loss(h_txt_all[0], h_txt_all[4])
        recon_loss = (
            img_recon_loss
            + txt_recon_loss
            + (img_cycle_loss + txt_cycle_loss) * opt.args.lamda1
            + (img_cycle_recon_loss + txt_cycle_recon_loss)
        )

        img_txt = torch.mm(h2_img_feat, h2_txt_feat.transpose(0, 1))
        con_loss = (
            torch.diagonal(img_txt).add(-1).pow(2).mean()
            + off_diagonal(img_txt).pow(2).mean()
        )

        loss = recon_loss + opt.args.lamda2 * con_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("{} loss: {}".format(epoch, loss))

        kmeans = KMeans(config["n_clusters"], max_iter=1000, tol=5e-5, n_init=20).fit(
            h_fuse.data.cpu().numpy()
        )
        train_metrics = calculate_metrics(labels, kmeans.labels_)
        print(">Train", METRIC_PRINT.format(*train_metrics))


if __name__ == "__main__":
    config = dict()
    if opt.args.dataset == "wikipedia":
        config["img_input_dim"] = 2048
        config["txt_input_dim"] = 2048
        config["n_clusters"] = 10
        config["img_hiddens"] = [1024, 256, 128]
        config["txt_hiddens"] = [1024, 256, 128]
        config["img2txt_hiddens"] = [128, 256, 128]
        config["txt2img_hiddens"] = [128, 256, 128]
        # if the data include corresponding filename for each sample feature
        config["has_filename"] = True
    elif opt.args.dataset == "nuswide":
        config["img_input_dim"] = 1000
        config["txt_input_dim"] = 1000
        config["n_clusters"] = 10
        config["img_hiddens"] = [512, 256, 128]
        config["txt_hiddens"] = [512, 128]
        config["img2txt_hiddens"] = [128, 128]
        config["txt2img_hiddens"] = [128, 128]
        config["has_filename"] = False
    config["batchnorm"] = True
    config["cuda"] = use_cuda
    config["device"] = device

    check_dir_exist(opt.args.cpt_dir)
    torch.manual_seed(opt.args.seed)
    torch.cuda.manual_seed(opt.args.seed)

    model = IDMMC(opt.args, config, opt.args.dm2c_cptpath)
    print(model)
    if use_cuda:
        model.cuda()

    train_data = MFeatDataSet(
        file_mat=os.path.join(opt.args.data_dir, "train_file.mat"),
        has_filename=config["has_filename"],
    )
    time_start = time.time()
    dataloader = DataLoader(dataset=train_data, batch_size=1910, shuffle=False)
    print(dataloader)
    for step, (ids, feats, modalitys, labels) in enumerate(dataloader):
        feats, modalitys = feats.cuda(), modalitys.cuda()
        labels = labels.numpy()
        train_IDMMC(model, feats, modalitys, labels)

    time_end = time.time()
    time = (time_end - time_start) / 60
    print("time:{}".format(time))
