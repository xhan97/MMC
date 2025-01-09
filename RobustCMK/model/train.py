import torch
import torch.nn.functional as F
from sklearn.cluster import k_means
from utils.score import cluster_metric


def train(data_loader, model, criterion, optimizer, epoch, num_class, args, **params):
    device = torch.device(args.device)
    loss_all = 0
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
        feature, label = [feature[i].to(device) for i in range(num_view)], label.to(
            device
        )
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

        features = torch.cat(proj_feature, dim=0)

        (
            H,
            loss_all_batch,
            pos_avg_batch,
            neg_avg_batch,
            true_neg_avg_batch,
            false_neg_avg_batch,
        ) = criterion(
            features,
            pos_mask,
            neg_mask,
            true_neg_mask,
            false_neg_mask,
            m=args.margin,
        )
        loss_all += loss_all_batch.item()  # 总的损失
        pos_avg += (pos_avg_batch - pos_avg) / (batch_idx + 1)
        neg_avg += (neg_avg_batch - neg_avg) / (batch_idx + 1)
        true_neg_avg += (true_neg_avg_batch - true_neg_avg) / (batch_idx + 1)
        false_neg_avg += (false_neg_avg_batch - false_neg_avg) / (batch_idx + 1)

        # loss of extra downstream task

        optimizer.zero_grad()
        loss_all.backward()
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
    return (
        loss_all,
        loss_clu,
        pos_avg,
        neg_avg,
        true_neg_avg,
        false_neg_avg,
        acc,
        nmi,
        pur,
    )
