import random

import hdf5storage as hdf
import numpy as np
import argparse

import torch


def load_mat(path, key_feature="data", key_label="labels"):
    data = hdf.loadmat(path)
    feature = []
    num_view = len(data[key_feature])
    label = data[key_label].reshape((-1,))
    num_smp = label.size
    for v in range(num_view):
        tmp = data[key_feature][v][0].squeeze()
        feature.append(tmp)
    # 打乱样本
    rand_permute = np.random.permutation(num_smp)
    for v in range(num_view):
        feature[v] = feature[v][rand_permute]
    label = label[rand_permute]
    return feature, label


def load_outer_args():
    parser = argparse.ArgumentParser(description="robust cmk")
    #  MNIST10 ALOI_100 animal
    parser.add_argument("-seed", "--seed", type=int, default=16, help="random seed")
    parser.add_argument("-ds", "--dataset", type=str, default="BBCSport", help="the multi-view dataset")
    parser.add_argument("-m", "--margin", type=float, default=1.0,
                        help="default value is not used, will init during running")
    parser.add_argument("-mr", "--margin_rate", type=float, default=1.6, help="margin rate * baseline")
    parser.add_argument("-e", "--epoches", type=int, default=100, help="num of epoches")
    parser.add_argument("-b", "--base_dir", type=str, default="./data",
                        help="source dir of dataset")
    parser.add_argument("-d", "--proj_dimension", type=int, default=128, help="project all views to same dimension")
    parser.add_argument("-bt", "--batch_size", type=int, default=32, help="batch size of each iteration")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="learning rate of optimizer")
    parser.add_argument("-dv", "--device", type=str, default="cuda", help="dev used in training")
    parser.add_argument("-t", "--trade_off", type=float, default=0.1, help="balance between loss_con and loss_clu")
    parser.add_argument("-r", "--robust", action="store_false", default=True, help="use robust loss or cmk")
    args_default = parser.parse_args()
    return args_default


def init_torch(device:str, seed):
    # 随机数种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

