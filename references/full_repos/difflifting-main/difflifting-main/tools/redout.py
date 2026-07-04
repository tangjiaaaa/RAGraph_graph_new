"""Abstract base class for readout layers."""

from abc import abstractmethod

import topomodelx
import torch
import torch_geometric
from torch_geometric.utils import scatter


class AbstractZeroCellReadOut(torch.nn.Module):
    r"""Readout layer for GNNs that operates on the batch level.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of the GNN model.
    out_channels : int
        Number of output channels.
    task_level : str
        Task level for readout layer. Either "graph" or "node".
    pooling_type : str
        Pooling type for readout layer. Either "max", "sum" or "mean".
    **kwargs : dict
        Additional arguments.
    """

    def __init__(
        self,
        hidden_dim: int,
        out_channels: int,
        task_level: str,
        pooling_type: str = "sum",
        **kwargs,
    ):
        super().__init__()

        self.linear = torch.nn.Linear(hidden_dim, out_channels)
        assert task_level in ["graph", "node"], "Invalid task_level"
        self.task_level = task_level

        assert pooling_type in ["max", "sum", "mean"], "Invalid pooling_type"
        self.pooling_type = pooling_type

    def __repr__(self):
        return f"{self.__class__.__name__}(task_level={self.task_level}, pooling_type={self.pooling_type})"

    def __call__(
        self, model_out: dict, batch: torch_geometric.data.Data=None
    ) -> dict:
        """Readout logic based on model_output.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output.
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        model_out = self.forward(model_out, batch)
        if batch is not None:
            model_out["logits"] = self.compute_logits(
                model_out["x_0"], batch["batch_0"]
            )
        else:
            model_out["logits"] = self.compute_logits(
                model_out["x_0"]
            )

        return model_out

    def compute_logits(self, x, batch=None):
        r"""Compute logits based on the readout layer.

        Parameters
        ----------
        x : torch.Tensor
            Node embeddings.
        batch : torch.Tensor
            Batch index tensor.

        Returns
        -------
        torch.Tensor
            Logits tensor.
        """
        if self.task_level == "graph":
            x = scatter(x, batch, dim=0, reduce=self.pooling_type)
            #print("PASSOOUUUU\n")

        return self.linear(x)

    @abstractmethod
    def forward(self, model_out: dict, batch: torch_geometric.data.Data=None):
        r"""Forward pass.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output.
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.
        """

class PropagateSignalDown(AbstractZeroCellReadOut):
    r"""Propagate signal down readout layer.

    This readout layer propagates the signal from cells of a certain order to the cells of the lower order.

    Parameters
    ----------
    **kwargs : dict
        Additional keyword arguments. It should contain the following keys:
        - num_cell_dimensions (int): Highest order of cells considered by the model.
        - hidden_dim (int): Dimension of the cells representations.
        - readout_name (str): Readout name.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.name = kwargs["readout_name"]
        self.dimensions = range(kwargs["num_cell_dimensions"] - 1, 0, -1)
        hidden_dim = kwargs["hidden_dim"]

        for i in self.dimensions:
            setattr(
                self,
                f"agg_conv_{i}",
                topomodelx.base.conv.Conv(
                    hidden_dim, hidden_dim, aggr_norm=False
                ),
            )

            setattr(self, f"ln_{i}", torch.nn.LayerNorm(hidden_dim))

            setattr(
                self,
                f"projector_{i}",
                torch.nn.Linear(2 * hidden_dim, hidden_dim),
            )

    def forward(self, model_out: dict, batch: torch_geometric.data.Data=None):
        r"""Forward pass of the propagate signal down readout layer.

        The layer takes the embeddings of the cells of a certain order and applies a convolutional layer to them. Layer normalization is then applied to the features. The output is concatenated with the initial embeddings of the cells and the result is projected with the use of a linear layer to the dimensions of the cells of lower rank. The process is repeated until the nodes embeddings, which are the cells of rank 0, are reached.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output.
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        for i in self.dimensions:

            x_i = getattr(self, f"agg_conv_{i}")(
                model_out[f"x_{i}"], batch[f"incidence_{i}"]
            )

            x_i = getattr(self, f"ln_{i}")(x_i)
            model_out[f"x_{i-1}"] = getattr(self, f"projector_{i}")(
                torch.cat([x_i, model_out[f"x_{i-1}"]], dim=1)
            )

        return model_out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_cell_dimensions={len(self.dimensions)}, hidden_dim={self.hidden_dim}, readout_name={self.name}"


class DirectReadout(AbstractZeroCellReadOut):
    r"""
    Direct readout layer that skips propagation through dimensions.

    This readout layer computes logits directly from the `x_0` embeddings,
    bypassing signal propagation through higher-order cells.

    Parameters
    ----------
    **kwargs : dict
        Additional keyword arguments, including hidden_dim, out_channels,
        task_level, and pooling_type.
    """
    def __init__(self, **kwargs):
        # Pass all kwargs to the parent class
        super().__init__(**kwargs)

    def forward(self, model_out: dict, batch: torch_geometric.data.Data=None):
        r"""
        Forward pass for direct readout.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output.
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        # Directly use `x_0` embeddings to compute logits
        if batch is not None:
            model_out["logits"] = self.compute_logits(
                model_out["x_0"], batch["batch_0"]
            )
        else:
            model_out["logits"] = self.compute_logits(model_out["x_0"])
        return model_out
