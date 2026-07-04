import torch
import numpy as np
import scipy.sparse as sp
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


def process_tu_dataset(data, num_node_attributes):
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

    for graph in data_list:
        # 检查是否有 x、边、节点
        if not hasattr(graph, 'x') or graph.x is None:
            print("[跳过] 图没有特征，跳过")
            continue
        if graph.x.numel() == 0:
            print("[跳过] 图特征为空，跳过")
            continue
        if not hasattr(graph, 'edge_index') or graph.edge_index is None or graph.edge_index.numel() == 0:
            print("[跳过] 图没有边，跳过")
            continue
        if graph.x.shape[0] <= 2:
            print(f"[跳过] 图节点太少：{graph.x.shape[0]} 个，跳过")
            continue

        x = graph.x

        if labelnum is None:
            ft_size = x.size(1)
            labelnum = range(num_node_attributes, ft_size)

        f = x[:, num].cpu().numpy()
        l = x[:, labelnum].cpu().numpy()

        if np.isnan(f).any() or np.isnan(l).any():
            print("[跳过] 图中有 NaN 特征或标签，跳过")
            continue

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

    # 拼接特征和标签
    features = np.vstack(features_list)
    if not features_list:
        raise ValueError("No valid graphs left after filtering in process_tu_dataset")

    features = np.vstack(features_list)
    node_labels = np.vstack(labels_list)

    # 拼接邻接矩阵为 block-diagonal
    adj = sum(adjacency_blocks)
    adj = adj.tocsr()

    # 归一化邻接矩阵
    adj = normalize_adj(adj + sp.eye(adj.shape[0])).todense()

    # 转成 tensor
    features = torch.FloatTensor(features).cuda()
    adj = torch.FloatTensor(adj).cuda()
    node_labels = torch.FloatTensor(node_labels).cuda()

    return features, adj, node_labels


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
