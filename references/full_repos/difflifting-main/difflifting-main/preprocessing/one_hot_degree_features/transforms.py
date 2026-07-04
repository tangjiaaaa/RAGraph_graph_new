
import torch
import torch_geometric
from torch_geometric.utils import one_hot


import torch_geometric


class NodeDegrees(torch_geometric.transforms.BaseTransform):
    r"""A transform that calculates the node degrees of the input graph.

    Parameters
    ----------
    **kwargs : optional
        Parameters for the base transform.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "node_degrees"
        self.parameters = kwargs
        self.parameters["selected_fields"] = "edge_index"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.type!r}, parameters={self.parameters!r})"

    def forward(self, data: torch_geometric.data.Data):
        r"""Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data.
        """

        data = self.calculate_node_degrees(data, "edge_index")

        return data


    def calculate_node_degrees(
        self, data: torch_geometric.data.Data, field: str
    ) -> torch_geometric.data.Data:
        r"""Calculate the node degrees of the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.
        field : str
            The field to calculate the node degrees.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data.
        """

        # Get number of nodes
        if data.get("num_nodes", None):
            max_num_nodes = data["num_nodes"]
        else:
            max_num_nodes = data["x"].shape[0]
        degrees = (
            torch_geometric.utils.to_dense_adj(
                data[field],
                max_num_nodes=max_num_nodes,
            )
            .squeeze(0)
            .sum(1)
        )


        field_name = "node_degrees"

        data[field_name] = degrees.unsqueeze(1)
        return data


class OneHotDegreeFeatures(torch_geometric.transforms.BaseTransform):
    r"""Class for one hot degree features transform.

    A transform that adds the node degree as one hot encodings to the node features.

    Parameters
    ----------
    max_degree : int
        The maximum degree of the graph.
    degrees_fields : str
        The field containing the node degrees.
    features_fields : str
        The field containing the node features.
    cat : bool, optional
        If set to `True`, the one hot encodings are concatenated to the node
        features (default: False).
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(
        self,
        max_degree: int=135,
        degrees_fields: str="node_degrees",
        features_fields: str="x",
        cat: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.type = max_degree
        self.max_degree = max_degree
        self.degrees_field = degrees_fields
        self.features_field = features_fields
        self.cat = cat

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.type!r}, max_degree={self.max_degree}, degrees_field={self.deg_field!r}, features_field={self.features_fields!r})"

    def forward(self, data: torch_geometric.data.Data):
        r"""Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data.
        """
        assert data.edge_index is not None

        deg = data[self.degrees_field].to(torch.long)

        if len(deg.shape) == 2:
            deg = deg.squeeze(1)

        deg = one_hot(deg, num_classes=self.max_degree + 1)

        if self.cat:
            x = data[self.features_field]
            x = x.view(-1, 1) if x.dim() == 1 else x
            data[self.features_field] = torch.cat([x, deg.to(x.dtype)], dim=-1)
        else:
            data[self.features_field] = deg

        return data
