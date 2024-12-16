import torch
import torch.nn as nn
from mkkm.kernel import rbf_kernel


# copy from "Contrastive Multi-view Kernel Learning"
class CMKLoss(nn.Module):
    def __init__(self, kernel_options, device=torch.device('cpu')):
        super(CMKLoss, self).__init__()
        self.kernel_options = kernel_options
        self.device = device

    def forward(self, feature, pos_mask, neg_mask, true_neg_mask, false_neg_mask, **kwargs):
        # flatten features
        num_view, num_smp = len(feature), feature[0].shape[0]
        features = torch.cat(feature, dim=0)

        mask = pos_mask
        logits_mask = pos_mask + neg_mask

        # define Euclidean distance
        def EuDist2(fea_a, fea_b):
            num_smp = fea_a.shape[0]
            aa = torch.sum(fea_a * fea_a, 1, keepdim=True)
            bb = torch.sum(fea_b * fea_b, 1, keepdim=True)
            ab = torch.matmul(fea_a, fea_b.T)
            D = aa.repeat([1, num_smp]) + bb.repeat([1, num_smp]) - 2 * ab
            return D

        # compute kernels
        if self.kernel_options['type'] == 'rbf':
            K = rbf_kernel(features, gamma=self.kernel_options['gamma'])
        elif self.kernel_options['type'] == 'Gaussian':
            D = EuDist2(features, features)
            K = torch.exp(-D / (2 * self.kernel_options['t'] ** 2))
        elif self.kernel_options['type'] == 'Linear':
            K = torch.matmul(features, features.T)
        elif self.kernel_options['type'] == 'Polynomial':
            K = torch.pow(self.kernel_options['a'] * torch.matmul(features, features.T) + self.kernel_options['b'],
                          self.kernel_options['d'])
        elif self.kernel_options['type'] == 'Sigmoid':
            K = torch.tanh(self.kernel_options['d'] * torch.matmul(features, features.T) + self.kernel_options['c'])
        elif self.kernel_options['type'] == 'Cauchy':
            D = EuDist2(features, features)
            K = 1 / (D / self.kernel_options['sigma'] + 1)
        else:
            raise NotImplementedError
        # loss of contrastive learning
        logits = torch.exp(K)
        log_prob = torch.log(logits) - torch.log((logits * logits_mask).sum(1, keepdim=True))
        mean_log_prob_pos = - (mask * log_prob).sum(1) / mask.sum(1)
        loss_con = mean_log_prob_pos.mean()

        with torch.no_grad():
            pos_avg = torch.sum(K * pos_mask) / torch.sum(pos_mask)
            neg_avg = torch.sum(K * neg_mask) / torch.sum(neg_mask)
            true_neg_avg = torch.sum(K * true_neg_mask) / torch.sum(true_neg_mask)
            false_neg_avg = torch.sum(K * false_neg_mask) / torch.sum(false_neg_mask)
        return loss_con, pos_avg.item(), neg_avg.item(), true_neg_avg.item(), false_neg_avg.item()


class RCMKLoss(nn.Module):
    def __init__(self, kernel_options: dict, device=torch.device('cpu')):
        super(RCMKLoss, self).__init__()
        self.kernel_options = kernel_options
        self.device = device

    def forward(self, feature, pos_mask, neg_mask, true_neg_mask, false_neg_mask, **kwargs):
        m = kwargs["m"]
        features = torch.cat(feature, dim=0)

        # define Euclidean distance
        def EuDist2(fea_a, fea_b):
            return torch.cdist(fea_a, fea_b, p=2)

        # compute kernels
        if self.kernel_options['type'] == 'rbf':
            K = rbf_kernel(features, gamma=self.kernel_options['gamma'])
        elif self.kernel_options['type'] == 'Gaussian':
            D = EuDist2(features, features)
            K = torch.exp(-D / (2 * self.kernel_options['t'] ** 2))
        elif self.kernel_options['type'] == 'Linear':
            K = torch.matmul(features, features.T)
        elif self.kernel_options['type'] == 'Polynomial':
            K = torch.pow(self.kernel_options['a'] * torch.matmul(features, features.T) + self.kernel_options['b'],
                          self.kernel_options['d'])
        elif self.kernel_options['type'] == 'Sigmoid':
            K = torch.tanh(self.kernel_options['d'] * torch.matmul(features, features.T) + self.kernel_options['c'])
        elif self.kernel_options['type'] == 'Cauchy':
            D = EuDist2(features, features)
            K = 1 / (D / self.kernel_options['sigma'] + 1)
        else:
            raise NotImplementedError
        # loss of contrastive learning
        dist = torch.exp(-K)
        # dist = EuDist2(features, features)
        pos_loss = dist ** 2
        # neg_loss = 1 / m * torch.clamp((m - dist) * torch.sqrt(dist), min=0) ** 2
        neg_loss = (1 / m) * torch.pow(torch.clamp(torch.pow(dist, 0.5) * (m - dist), min=0.0), 2)
        loss_con = torch.sum(pos_loss * pos_mask + neg_loss * neg_mask) / torch.sum(pos_mask + neg_mask)

        with torch.no_grad():
            pos_avg = torch.sum(dist * pos_mask) / torch.sum(pos_mask)
            neg_avg = torch.sum(dist * neg_mask) / torch.sum(neg_mask)
            true_neg_avg = torch.sum(dist * true_neg_mask) / torch.sum(true_neg_mask)
            false_neg_avg = torch.sum(dist * false_neg_mask) / torch.sum(false_neg_mask)
        return loss_con, pos_avg.item(), neg_avg.item(), true_neg_avg.item(), false_neg_avg.item()
