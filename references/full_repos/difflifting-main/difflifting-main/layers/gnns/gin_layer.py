import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINConv
from layers.gnns.gnn_interface import GNNFactoryInterface


class GINLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        activation,
        batch_norm,
        mlp_hidden_dim=None,
        residual=True,
        train_eps=False,
    ):
        super().__init__()

        if mlp_hidden_dim is None:
            mlp_hidden_dim = in_features

        self.activation = activation
        self.residual = residual
        self.batchnorm = nn.BatchNorm1d(out_features) if batch_norm else nn.Identity()
        gin_net = nn.Sequential(
            nn.Linear(in_features, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, out_features),
        )
        self.conv = GINConv(gin_net, train_eps=train_eps)

    def forward(self, x, edge_index, **kwargs):
        h = self.conv(x, edge_index)
        h = self.batchnorm(h)
        if self.residual:
            h = h + x
        return h


class GinCreator(GNNFactoryInterface):
    def __init__(self, hidden_dim, batch_norm):
        self.hidden_dim = hidden_dim
        self.batch_norm = batch_norm

    def return_gnn_instance(self, is_last=False):
        return GINLayer(
            in_features=self.hidden_dim,
            out_features=self.hidden_dim,
            activation=nn.Identity() if is_last else F.relu,
            batch_norm=self.batch_norm,
        )