from torch_geometric.transforms import BaseTransform, to_sparse_tensor
import torch

class ProjectionSum(BaseTransform):
    r"""Lift r-cell features to r+1-cells by projection."""

    def __init__(self, **kwargs):
        super().__init__()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def lift_features(self, data):
        r"""Project r-cell features of a graph to r+1-cell structures."""
        keys = sorted(
            [key.split("_")[1] for key in data if ("incidence" in key and "-" not in key)]
        )
        for elem in keys:
            if f"x_{elem}" not in data:
                idx_to_project = 0 if elem == "hyperedges" else int(elem) - 1
                data["x_" + elem] = torch.matmul(
                    abs(data["incidence_" + elem].t()).float(), data[f"x_{idx_to_project}"].float()
                )
        data["x_0"] = data["x_0"].float()
        return data

    def forward(self, data):
        r"""Apply the lifting to the input data."""
        data = self.lift_features(data)
        return data