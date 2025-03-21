import torch.nn as nn
import torch.nn.functional as F


# Model definitions
class FCNet(nn.Module):
    """Fully connected network for feature extraction."""

    def __init__(self, feat_dims, latent_dim=64, normalize=True):
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
