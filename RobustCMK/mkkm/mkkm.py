import torch


def kkm(kdata, k):
    """
    :param kdata: kernel matrix
    :return: H, obj
    """
    # 对核矩阵进行特征值分解
    sigma, vector = torch.linalg.eig(kdata)
    # 按照特征值大小进行排序，取前k个
    ind = torch.argsort(torch.real(sigma), descending=True)
    H = torch.real(vector[:, ind[:k]])
    # 返回H 和 相应的目标函数
    obj = torch.trace(kdata) - torch.trace(H.T @ kdata @ H)
    return H, obj


def mkkm(mkdata, k, theta=None, fix_theta=True, eps=1e-6, max_iter=250):
    """
    :param max_iter:
    :param eps:
    :param theta:
    :param mkdata: multi view kernel matrices n:n:v
    :param k: num of class
    :param fix_theta: fix view weights or not
    :return: H, obj, weight
    """
    num_view = len(mkdata)
    if theta is None:
        theta = torch.ones((num_view,), device=mkdata.device) / num_view
    if not fix_theta:
        err = 1
        while err > eps and max_iter > 0:
            kdata = torch.sum(mkdata * theta, dim=2)
            H, obj = kkm(kdata, k)
            # 计算系数
            coef = torch.zeros((num_view,), device=mkdata.device)
            for i in range(num_view):
                coef[i] = torch.trace(mkdata[:, :, i]) - torch.trace(H.T @ mkdata[:, :, i] @ H)
            theta_new = 1 / coef
            theta_new = theta_new / torch.sum(theta_new)
            err = torch.sum(theta - theta_new)
            theta = theta_new
            max_iter -= 1
    kdata = torch.sum(mkdata * theta[:, None, None], dim=0)
    H, obj = kkm(kdata, k)
    return H, obj, theta
