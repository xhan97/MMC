import torch
from torch.utils.data import Dataset
from utils.common import load_mat


class MultiviewDataset(Dataset):
    def __init__(self, root: str, key_feature="data", key_label="labels"):
        """
        :param root: path of dataset
        """
        # load_scene --- customize load_<dataset> function if not consistent
        data, labels = load_mat(root, key_feature=key_feature, key_label=key_label)
        self.num_view = len(data)
        d_view = [0] * self.num_view
        for i in range(self.num_view):
            data[i] = torch.as_tensor(data[i], dtype=torch.float32)
            max_value, _ = torch.max(data[i], dim=0, keepdim=True)
            min_value, _ = torch.min(data[i], dim=0, keepdim=True)
            data[i] = (data[i] - min_value) / (max_value - min_value + 1e-12)
            d_view[i] = data[i].shape[1]
        self.data = data
        self.d_view = d_view
        self.labels = torch.as_tensor(labels, dtype=torch.int64).view((-1,))
        self.num_class = len(torch.unique(self.labels))

    def __getitem__(self, index):
        item = []
        for i in range(self.num_view):
            item.append(torch.as_tensor(self.data[i][index]))
        return item, self.labels[index]

    def __len__(self):
        return len(self.labels)
