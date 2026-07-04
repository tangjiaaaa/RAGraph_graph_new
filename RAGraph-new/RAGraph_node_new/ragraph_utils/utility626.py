import torch
import numpy as np
import scipy.sparse as sp
import networkx as nx
from ragraph_utils.complex import Cochain, Complex
from torch_geometric.data import Data, Batch
def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def process_tu_dataset626(data, num_node_attributes):
    if isinstance(data, Data):
        data_list = [data]
    elif isinstance(data, Batch):
        data_list = data.to_data_list()
    elif isinstance(data, list):
        data_list = data
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    features_list = []
    labels_list = []
    adjacency_blocks = []

    num = range(num_node_attributes)
    labelnum = None
    total_nodes = 0
    all_edges = []
    all_ring_rows = []
    all_ring_cols = []
    for graph in data_list:
        if not hasattr(graph, 'x') or graph.x is None:
            raise ValueError(f"Invalid graph object without features: {type(graph)}")

        x = graph.x
        if labelnum is None:
            ft_size = x.size(1)
            labelnum = range(num_node_attributes, ft_size)

        f = x[:, num].cpu().numpy()
        l = x[:, labelnum].cpu().numpy()

        features_list.append(f)
        labels_list.append(l)

        edge_index = graph.edge_index.cpu().numpy()
        num_nodes = f.shape[0]

        row, col = edge_index
        data_val = np.ones(len(row))
        coo = sp.coo_matrix((data_val, (row, col)), shape=(num_nodes, num_nodes))

        coo_shifted = sp.coo_matrix(
            (data_val, (row + total_nodes, col + total_nodes)),
            shape=(total_nodes + num_nodes, total_nodes + num_nodes)
        )
        adjacency_blocks.append(coo_shifted)
        total_nodes += num_nodes
        # 提取环结构
        G = nx.Graph()
        G.add_edges_from(edge_index.T.tolist())
        rings = nx.cycle_basis(G)
        for ring_id, cycle in enumerate(rings):
            if len(cycle) < 3 or len(cycle) > 6:
                continue
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                edge = tuple(sorted([u, v]))
                try:
                    edge_idx = all_edges.index(list(edge))
                except ValueError:
                    continue
                all_ring_rows.append(edge_idx)
                all_ring_cols.append(len(all_ring_cols))
    # 拼接特征和标签
    features = np.vstack(features_list)
    node_labels = np.vstack(labels_list)

    # 拼接邻接矩阵
    adj = sum(adjacency_blocks)
    adj = adj.tocsr()

    # 归一化邻接矩阵
    adj = normalize_adj(adj + sp.eye(adj.shape[0])).todense()

    # 转成 tensor
    features = torch.FloatTensor(features).cuda()
    adj = torch.FloatTensor(adj).cuda()
    node_labels = torch.FloatTensor(node_labels).cuda()
    # 创建 complex 对象
    edge_index = torch.tensor(np.array(all_edges).T, dtype=torch.long)
    v_cochain = Cochain(dim=0, x=features, y=node_labels)
    e_cochain = Cochain(dim=1, x=torch.ones((edge_index.shape[1], 1)), boundary_index=edge_index)

    if all_ring_rows:
        two_cell_boundary = torch.tensor([all_ring_rows, all_ring_cols], dtype=torch.long)
        two_cell_cochain = Cochain(dim=2, boundary_index=two_cell_boundary)
    else:
        two_cell_cochain = None

    complex_obj = Complex(v_cochain=v_cochain, e_cochain=e_cochain, two_cell_cochain=two_cell_cochain)
    return features, adj, node_labels, complex_obj


# Process a (subset of) a TU dataset into standard form
# def process_tu_dataset(data, num_node_attributes):
#
#     nb_graphs = data.num_graphs
#     ft_size = data.num_features
#
#     num = range(num_node_attributes)
#
#     labelnum=range(num_node_attributes, ft_size)
#
#     for g in range(nb_graphs):
#         if g == 0:
#             features = data[g].x[:, num]
#             rawlabels = data[g].x[:, labelnum]
#             e_ind = data[g].edge_index
#             coo = sp.coo_matrix((np.ones(e_ind.shape[1]), (e_ind[0, :], e_ind[1, :])),
#                                 shape=(features.shape[0], features.shape[0]))
#             adjacency = coo.todense()
#         else:
#             tmpfeature = data[g].x[:, num]
#             features = np.row_stack((features, tmpfeature))
#             tmplabel = data[g].x[:, labelnum]
#             rawlabels = np.row_stack((rawlabels, tmplabel))
#             e_ind = data[g].edge_index
#             coo = sp.coo_matrix((np.ones(e_ind.shape[1]), (e_ind[0, :], e_ind[1, :])),
#                                 shape=(tmpfeature.shape[0], tmpfeature.shape[0]))
#             # print("coo",coo)
#             tmpadj = coo.todense()
#             zero = np.zeros((adjacency.shape[0], tmpfeature.shape[0]))
#             tmpadj1 = np.column_stack((adjacency, zero))
#             tmpadj2 = np.column_stack((zero.T, tmpadj))
#             adjacency = np.row_stack((tmpadj1, tmpadj2))
#
#
#     node_labels =rawlabels
#     adj = sp.csr_matrix(adjacency)
#
#     # postprocess
#     adj = normalize_adj(adj + sp.eye(adj.shape[0])).todense()
#     features = torch.FloatTensor(features).cuda()
#     adj = torch.FloatTensor(adj).cuda()
#     node_labels = torch.FloatTensor(node_labels).cuda()
#
#
#     return features, adj, node_labels
