import numpy as np
import random
import torch
import torch.nn.functional as F
from mkkm.mkkm import mkkm
from sklearn.cluster import k_means
from utils.score import cluster_metric


def train(
        data_loader,
        model,
        criterion,
        optimizer,
        epoch,
        kernel_func,
        num_class,
        args,
        **params
):
    device = torch.device(args.device)
    loss_con = 0
    loss_clu = 0
    pos_avg = 0
    neg_avg = 0
    true_neg_avg = 0
    false_neg_avg = 0

    acc, nmi, pur = 0, 0, 0
    for batch_idx, (feature, label) in enumerate(data_loader):

        num_view = len(feature)
        num_smp = len(feature[0])
        num_ins = num_view * num_smp
        feature, label = [feature[i].to(device) for i in range(num_view)], label.to(device)
        proj_feature = model(feature)
        pos_mask = torch.eye(num_smp, device=device)
        pos_mask = pos_mask.repeat(num_view, num_view)
        neg_mask = 1 - pos_mask
        idx = torch.arange(0, num_ins, device=device)
        pos_mask[idx, idx] = 0

        label_vec = label.repeat(num_view, 1).view(1, -1)
        identity_matrix = torch.as_tensor(label_vec == label_vec.T, device=device)
        true_neg_mask = torch.where(identity_matrix, 0, 1)
        false_neg_mask = neg_mask - true_neg_mask
        loss_con_batch, pos_avg_batch, neg_avg_batch, true_neg_avg_batch, false_neg_avg_batch \
            = criterion(proj_feature, pos_mask, neg_mask, true_neg_mask, false_neg_mask, m=args.margin)

        loss_con += loss_con_batch.item()  # 总的损失
        pos_avg += (pos_avg_batch - pos_avg) / (batch_idx + 1)
        neg_avg += (neg_avg_batch - neg_avg) / (batch_idx + 1)
        true_neg_avg += (true_neg_avg_batch - true_neg_avg) / (batch_idx + 1)
        false_neg_avg += (false_neg_avg_batch - false_neg_avg) / (batch_idx + 1)

        multi_kmatrix = []
        for i in range(num_view):
            kmatrix = kernel_func(proj_feature[i], **params)
            multi_kmatrix.append(kmatrix)
        kernel = torch.stack(multi_kmatrix, dim=0).mean(0)
        val, vec = torch.linalg.eig(kernel.detach())
        _, ind = torch.sort(torch.real(val.detach().cpu()), descending=True)
        H = torch.real(vec[:, ind[:num_class]])

        # loss of extra downstream task
        loss_clu_batch = (torch.trace(kernel) - torch.trace(torch.linalg.multi_dot([H.T, kernel, H]))) / num_smp

        # get normalized H for later validation
        H = F.normalize(H).detach().cpu().numpy()
        if epoch < args.epoches // 3:
            loss_total_batch = loss_con_batch
        else:
            loss_total_batch = loss_con_batch + loss_clu_batch * args.trade_off
        loss_clu += loss_clu_batch.item()
        loss_con += loss_con_batch.item()

        if epoch > 0:
            optimizer.zero_grad()
            loss_total_batch.backward()
            optimizer.step()

        # k_means
        centroid, y_pred, _ = k_means(H, num_class, n_init="auto")
        y_true = label.cpu().numpy()
        rate = len(y_true) / args.batch_size
        acc_bt, nmi_bt, pur_bt = cluster_metric(y_true, y_pred)
        acc += (acc_bt - acc) * rate / (batch_idx + rate)
        nmi += (nmi_bt - nmi) * rate / (batch_idx + rate)
        pur += (pur_bt - pur) * rate / (batch_idx + rate)
    if epoch == 0:
        args.margin = (pos_avg + neg_avg) / 2 * args.margin_rate
    return loss_con, loss_clu, pos_avg, neg_avg, true_neg_avg, false_neg_avg, acc, nmi, pur
