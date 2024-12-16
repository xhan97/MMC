import argparse
import logging as log
import os
import sys
import torch
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from mkkm.kernel import rbf_kernel
from model.loss import RCMKLoss, CMKLoss
# from model.loss import sim_to_dist
from model.projection import LinearProjection
from model.train import train
from utils.dataset.custom_dataset import MultiviewDataset
from utils.common import load_outer_args, init_torch


def main(args):

    if not torch.cuda.is_available():
        args.device = "cpu"
    init_torch(args.device, args.seed)
    base_dir = args.base_dir
    dataset = args.dataset
    path = os.path.join(base_dir, dataset + ".mat")
    # data, labels = load_mat(args.dataset)
    multi_data = MultiviewDataset(path, "data", "labels")
    num_view = multi_data.num_view
    num_class = multi_data.num_class
    d_view = multi_data.d_view
    num_smp = len(multi_data)
    drop_last = False
    if num_smp % args.batch_size < num_class * 2:
        drop_last = True
    data_loader = DataLoader(multi_data, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=drop_last)
    model = LinearProjection(args.proj_dimension, d_view).to(args.device)
    metric_func = rbf_kernel
    params = {"gamma": 1 / args.proj_dimension}
    rcmk_loss = RCMKLoss(dict(type='rbf', gamma=1 / args.proj_dimension), device=args.device)
    cmk_loss = CMKLoss(dict(type='rbf', gamma=1 / args.proj_dimension), device=args.device)
    # optimizer = SGD(model.parameters(), lr=args.learning_rate, weight_decay=0, momentum=0.9)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=0)
    if not args.robust:
        criterion = cmk_loss
    else:
        criterion = rcmk_loss
    acc, nmi, pur = 0, 0, 0
    for epoch in range(args.epoches):
        model.train()
        loss_con, loss_clu, pos_avg, neg_avg, true_neg_avg, false_neg_avg, acc, nmi, pur \
            = train(data_loader, model, criterion, optimizer, epoch, metric_func, num_class, args, **params)
        # 打印每一轮的信息
        log.info(f"epoch: {epoch}, loss_con: {loss_con}, loss_clu: {loss_clu}, pos_avg: {pos_avg}, "
                 f"neg_avg: {neg_avg}, true_neg_avg: {true_neg_avg}, false_neg_avg: {false_neg_avg}")
        log.info(f"------ acc: {acc}, nmi: {nmi}, pur: {pur}")
    return acc, nmi, pur


if __name__ == "__main__":
    # 日志部分
    log.basicConfig(stream=sys.stdout, level=log.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%m/%d %I:%M:%S %p")
    args_default = load_outer_args()
    main(args_default)
