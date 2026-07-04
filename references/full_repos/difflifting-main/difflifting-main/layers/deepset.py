import torch
import torch.nn as nn
from torch_scatter import scatter


class DeepSetLayer(nn.Module):
    """Simple equivariant deep set layer."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )

        self.external_mlp = nn.Sequential(nn.Linear(out_dim, out_dim))

    def forward(self, x, batch):
        x = self.mlp(x)
        x = scatter(x, batch, dim=0, reduce="mean")
        x = self.external_mlp(x)
        return x

