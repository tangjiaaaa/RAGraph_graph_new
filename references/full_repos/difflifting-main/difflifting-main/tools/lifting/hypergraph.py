"""This module implements the HypergraphKNNLifting class."""

import torch
import torch_geometric

from tools.lifting.lifting import GraphLifting


class Graph2HypergraphLifting(GraphLifting):
    r"""Abstract class for lifting graphs to hypergraphs.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "graph2hypergraph"

class HypergraphKNNLifting(Graph2HypergraphLifting):
    r"""Lift graphs to hypergraph domain by considering k-nearest neighbors.

    Parameters
    ----------
    k_value : int, optional
        The number of nearest neighbors to consider. Must be positive. Default is 1.
    loop : bool, optional
        If True the hyperedges will contain the node they were created from.
    **kwargs : optional
        Additional arguments for the class.

    Raises
    ------
    ValueError
        If k_value is less than 1.
    TypeError
        If k_value is not an integer or if loop is not a boolean.
    """

    def __init__(self, k_value=1, loop=True, **kwargs):
        super().__init__(**kwargs)

        # Validate k_value
        if not isinstance(k_value, int):
            raise TypeError("k_value must be an integer")
        if k_value < 1:
            raise ValueError("k_value must be greater than or equal to 1")

        # Validate loop
        if not isinstance(loop, bool):
            raise TypeError("loop must be a boolean")

        self.k = k_value
        self.loop = loop
        self.transform = torch_geometric.transforms.KNNGraph(self.k, self.loop)

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lift a graph to hypergraph by considering k-nearest neighbors.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        num_nodes = data.x.shape[0]
        data.pos = data.x
        num_hyperedges = num_nodes
        incidence_1 = torch.zeros(num_nodes, num_nodes)
        data_lifted = self.transform(data)
        # check for loops, since KNNGraph is inconsistent with nodes with equal features
        if self.loop:
            for i in range(num_nodes):
                if not torch.any(
                    torch.all(
                        data_lifted.edge_index == torch.tensor([[i, i]]).T,
                        dim=0,
                    )
                ):
                    connected_nodes = data_lifted.edge_index[
                        0, data_lifted.edge_index[1] == i
                    ]
                    dists = torch.sqrt(
                        torch.sum(
                            (
                                data.pos[connected_nodes]
                                - data.pos[i].unsqueeze(0) ** 2
                            ),
                            dim=1,
                        )
                    )
                    furthest = torch.argmax(dists)
                    idx = torch.where(
                        torch.all(
                            data_lifted.edge_index
                            == torch.tensor(
                                [[connected_nodes[furthest], i]]
                            ).T,
                            dim=0,
                        )
                    )[0]
                    data_lifted.edge_index[:, idx] = torch.tensor([[i, i]]).T

        incidence_1[data_lifted.edge_index[1], data_lifted.edge_index[0]] = 1
        incidence_1 = torch.Tensor(incidence_1).to_sparse_coo()
        return {
            "incidence_1": incidence_1,
            "num_hyperedges": num_hyperedges,
            "x_0": data.x,
        }

"""This module implements the k-hop lifting of graphs to hypergraphs."""



class HypergraphKHopLifting(Graph2HypergraphLifting):
    r"""Lift graph to hypergraphs by considering k-hop neighborhoods.

    The class transforms graphs to hypergraph domain by considering k-hop neighborhoods of
    a node. This lifting extracts a number of hyperedges equal to the number of
    nodes in the graph.

    Parameters
    ----------
    k_value : int, optional
        The number of hops to consider. Default is 1.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, k_value=1, **kwargs):
        super().__init__(**kwargs)
        self.k = k_value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(k={self.k!r})"

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lift a graphs to hypergraphs by considering k-hop neighborhoods.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        # Check if data has instance x:
        if hasattr(data, "x") and data.x is not None:
            num_nodes = data.x.shape[0]
        else:
            num_nodes = data.num_nodes

        incidence_1 = torch.zeros(num_nodes, num_nodes)
        edge_index = torch_geometric.utils.to_undirected(data.edge_index)

        # Detect isolated nodes
        isolated_nodes = [
            i for i in range(num_nodes) if i not in edge_index[0]
        ]
        if len(isolated_nodes) > 0:
            # Add completely isolated nodes to the edge_index
            edge_index = torch.cat(
                [
                    edge_index,
                    torch.tensor(
                        [isolated_nodes, isolated_nodes], dtype=torch.long
                    ),
                ],
                dim=1,
            )

        for n in range(num_nodes):
            neighbors, _, _, _ = torch_geometric.utils.k_hop_subgraph(
                n, self.k, edge_index
            )
            incidence_1[n, neighbors] = 1

        num_hyperedges = incidence_1.shape[1]
        incidence_1 = torch.Tensor(incidence_1).to_sparse_coo()
        return {
            "incidence_1": incidence_1,
            "num_hyperedges": num_hyperedges,
            "x_0": data.x,
        }