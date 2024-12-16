import torch
import torch.nn as nn


class LinearProjection(nn.Module):
    def __init__(self, d_proj, d_view, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_proj = d_proj
        self.d_view = d_view
        self.num_view = len(d_view)
        self.net_view = nn.ModuleList()
        for i in range(self.num_view):
            self.net_view.append(nn.Linear(d_view[i], d_proj))

    def forward(self, feature):
        """
        :param feature:
        :return: proj_feature 投影后的特征
        """
        proj_feature = []
        nets = self.net_view
        # 将各视图特征代入代入各自的proj_model
        for i in range(self.num_view):
            net = nets[i]
            tmp_feature = net(feature[i])
            proj_feature.append(net(feature[i]))
        return proj_feature
