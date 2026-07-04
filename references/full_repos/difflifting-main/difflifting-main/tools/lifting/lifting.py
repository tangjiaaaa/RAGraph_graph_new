"""Abstract class for lifting graphs to simplicial complexes."""

import networkx as nx
import torch
import torch_geometric
from networkx.classes import to_undirected
from topomodelx.utils.sparse import from_sparse
from toponetx.classes import SimplicialComplex
from toponetx.classes import CellComplex

import numpy as np
from torch_geometric.utils import is_undirected

from tools.lifting.abstract_lifting import AbstractLifting
from tools.lifting.utils import generate_zero_sparse_connectivity, select_neighborhoods_of_interest


def get_complex_connectivity(
    complex, max_rank, neighborhoods=None, signed=False
):
    """Get the connectivity matrices for the complex.

    Parameters
    ----------
    complex : toponetx.CellComplex or toponetx.SimplicialComplex
        Cell complex.
    max_rank : int
        Maximum rank of the complex.
    neighborhoods : list, optional
        List of neighborhoods of interest.
    signed : bool, optional
        If True, returns signed connectivity matrices.

    Returns
    -------
    dict
        Dictionary containing the connectivity matrices.
    """
    practical_shape = list(
        np.pad(list(complex.shape), (0, max_rank + 1 - len(complex.shape)))
    )
    connectivity = {}
    for rank_idx in range(max_rank + 1):
        for connectivity_info in [
            "incidence",
            "down_laplacian",
            "up_laplacian",
            "adjacency",
            "coadjacency",
            "hodge_laplacian",
        ]:
            try:
                connectivity[f"{connectivity_info}_{rank_idx}"] = from_sparse(
                    getattr(complex, f"{connectivity_info}_matrix")(
                        rank=rank_idx, signed=signed
                    )
                )
            except ValueError:
                if connectivity_info == "incidence":
                    connectivity[f"{connectivity_info}_{rank_idx}"] = (
                        generate_zero_sparse_connectivity(
                            m=practical_shape[rank_idx - 1],
                            n=practical_shape[rank_idx],
                        )
                    )
                else:
                    connectivity[f"{connectivity_info}_{rank_idx}"] = (
                        generate_zero_sparse_connectivity(
                            m=practical_shape[rank_idx],
                            n=practical_shape[rank_idx],
                        )
                    )
    if neighborhoods is not None:
        connectivity = select_neighborhoods_of_interest(
            connectivity, neighborhoods
        )
    connectivity["shape"] = practical_shape
    return connectivity

class GraphLifting(AbstractLifting):
    r"""Abstract class for lifting graph topologies to other domains.

    Parameters
    ----------
    feature_lifting : str, optional
        The feature lifting method to be used. Default is 'ProjectionSum'.
    preserve_edge_attr : bool, optional
        Whether to preserve edge attributes. Default is False.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(
        self,
        feature_lifting="ProjectionSum",
        preserve_edge_attr=False,
        **kwargs,
    ):
        super().__init__(feature_lifting=feature_lifting, **kwargs)
        self.preserve_edge_attr = preserve_edge_attr

    def _data_has_edge_attr(self, data: torch_geometric.data.Data) -> bool:
        r"""Check if the input data object has edge attributes.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        bool
            Whether the data object has edge attributes.
        """
        return hasattr(data, "edge_attr") and data.edge_attr is not None

    def _generate_graph_from_data(
        self, data: torch_geometric.data.Data
    ) -> nx.Graph:
        r"""Generate a NetworkX graph from the input data object.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        nx.Graph
            The generated NetworkX graph.
        """
        # Check if data object have edge_attr, return list of tuples as [(node_id, {'features':data}, 'dim':1)] or ??
        nodes = [
            (n, dict(features=data.x[n], dim=0))
            for n in range(data.x.shape[0])
        ]

        if self.preserve_edge_attr and self._data_has_edge_attr(data):
            # In case edge features are given, assign features to every edge
            edge_index, edge_attr = (
                data.edge_index,
                (
                    data.edge_attr
                    if is_undirected(data.edge_index, data.edge_attr)
                    else to_undirected(data.edge_index, data.edge_attr)
                ),
            )
            edges = [
                (i.item(), j.item(), dict(features=edge_attr[edge_idx], dim=1))
                for edge_idx, (i, j) in enumerate(
                    zip(edge_index[0], edge_index[1], strict=False)
                )
            ]
            self.contains_edge_attr = True
        else:
            # If edge_attr is not present, return list list of edges
            edges = [
                (i.item(), j.item(), {})
                for i, j in zip(
                    data.edge_index[0], data.edge_index[1], strict=False
                )
            ]
            self.contains_edge_attr = False
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph



class Graph2SimplicialLifting(GraphLifting):
    r"""Abstract class for lifting graphs to simplicial complexes.

    Parameters
    ----------
    complex_dim : int, optional
        The maximum dimension of the simplicial complex to be generated. Default is 2.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, complex_dim=2, **kwargs):
        super().__init__(**kwargs)
        self.complex_dim = complex_dim
        self.type = "graph2simplicial"
        self.signed = kwargs.get("signed", False)

    def _get_lifted_topology(
        self, simplicial_complex: SimplicialComplex, graph: nx.Graph
    ) -> dict:
        r"""Return the lifted topology.

        Parameters
        ----------
        simplicial_complex : SimplicialComplex
            The simplicial complex.
        graph : nx.Graph
            The input graph.

        Returns
        -------
        dict
            The lifted topology.
        """
        lifted_topology = get_complex_connectivity(
            simplicial_complex,
            self.complex_dim,
            neighborhoods=self.neighborhoods,
            signed=self.signed,
        )
        lifted_topology["x_0"] = torch.stack(
            list(
                simplicial_complex.get_simplex_attributes(
                    "features", 0
                ).values()
            )
        )
        # If new edges have been added during the lifting process, we discard the edge attributes
        if self.contains_edge_attr and simplicial_complex.shape[1] == (
            graph.number_of_edges()
        ):
            lifted_topology["x_1"] = torch.stack(
                list(
                    simplicial_complex.get_simplex_attributes(
                        "features", 1
                    ).values()
                )
            )
        return lifted_topology
    
class Graph2CellLifting(GraphLifting):
    r"""Abstract class for lifting graphs to cell complexes.

    Parameters
    ----------
    complex_dim : int, optional
        The dimension of the cell complex to be generated. Default is 2.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, complex_dim=2, **kwargs):
        super().__init__(**kwargs)
        self.complex_dim = complex_dim
        self.type = "graph2cell"

    def _get_lifted_topology(
        self, cell_complex: CellComplex, graph: nx.Graph
    ) -> dict:
        r"""Return the lifted topology.

        Parameters
        ----------
        cell_complex : CellComplex
            The cell complex.
        graph : nx.Graph
            The input graph.

        Returns
        -------
        dict
            The lifted topology.
        """
        lifted_topology = get_complex_connectivity(
            cell_complex, self.complex_dim, neighborhoods=self.neighborhoods
        )
        lifted_topology["x_0"] = torch.stack(
            list(cell_complex.get_cell_attributes("features", 0).values())
        )
        # If new edges have been added during the lifting process, we discard the edge attributes
        if self.contains_edge_attr and cell_complex.shape[1] == (
            graph.number_of_edges()
        ):
            lifted_topology["x_1"] = torch.stack(
                list(cell_complex.get_cell_attributes("features", 1).values())
            )
        return lifted_topology

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