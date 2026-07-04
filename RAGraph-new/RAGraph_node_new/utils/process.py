import os

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sys
import torch
import torch.nn as nn
from ragraph_utils.complex import Cochain, Complex
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_networkx

def parse_skipgram(fname):
    with open(fname) as f:
        toks = list(f.read().split())
    nb_nodes = int(toks[0])
    nb_features = int(toks[1])
    ret = np.empty((nb_nodes, nb_features))
    it = 2
    for i in range(nb_nodes):
        cur_nd = int(toks[it]) - 1
        it += 1
        for j in range(nb_features):
            cur_ft = float(toks[it])
            ret[cur_nd][j] = cur_ft
            it += 1
    return ret




def extract_rings(edge_index, num_nodes, max_ring=6):
    """提取无弦环（chordless cycles）作为2维胞腔"""
    G = nx.Graph()
    edges = edge_index.T.tolist()
    G.add_edges_from(edges)

    def is_cycle_edge(i1, i2, cycle):
        if i2 == i1 + 1 or (i1 == 0 and i2 == len(cycle) - 1):
            return True
        return False

    def is_chordless(cycle):
        for (i1, v1), (i2, v2) in itertools.combinations(enumerate(cycle), 2):
            if not is_cycle_edge(i1, i2, cycle) and G.has_edge(v1, v2):
                return False
        return True

    rings = set()
    for cycle in nx.simple_cycles(G.to_directed()):
        if 2 < len(cycle) <= max_ring and is_chordless(cycle):
            rings.add(tuple(sorted(cycle)))
    return rings

def construct_2cell_cochain(edge_index, num_nodes, max_ring=6):
    G = nx.Graph()
    edges = edge_index.T.tolist()
    G.add_edges_from(edges)

    # 构造边索引映射
    edge_dict = {}
    for idx, e in enumerate(edges):
        e = tuple(sorted(e))
        edge_dict[e] = idx

    # 获取三角形（或更高阶无弦环）
    rings = nx.cycle_basis(G)
    rows, cols = [], []
    for ring_id, cycle in enumerate(rings):
        if len(cycle) < 3 or len(cycle) > max_ring:
            continue
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
            edge = tuple(sorted([u, v]))
            if edge in edge_dict:
                rows.append(edge_dict[edge])
                cols.append(ring_id)

    if len(rows) == 0:
        return None
    boundary_index = torch.tensor([rows, cols], dtype=torch.long)
    return Cochain(dim=2, boundary_index=boundary_index)
# Process a (subset of) a TU dataset into standard form
def process_tu(data, class_num, max_ring=6):
    nb_graphs = data.num_graphs
    ft_size = data.num_features

    num = range(class_num)
    labelnum = range(class_num, ft_size)

    features_list = []
    labels_list = []
    adj_blocks = []
    offset = 0

    all_edges = []
    all_ring_rows = []
    all_ring_cols = []

    for g in range(nb_graphs):
        x = data.x
        edge_index = data.edge_index

        node_count = x.size(0)
        feature = x[:, num].numpy()
        label = x[:, labelnum].numpy()
        features_list.append(feature)
        labels_list.append(label)

        # 构造邻接矩阵（稀疏）
        coo = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
                            shape=(node_count, node_count))
        adj = coo.todense()
        adj_blocks.append(np.asarray(adj))  # 强制转为 numpy array


        # 记录全局边（需偏移）
        edge_np = edge_index.numpy().T + offset
        all_edges.extend(edge_np.tolist())

        # 提取2维环结构
        G = nx.Graph()
        G.add_edges_from(edge_np.tolist())
        rings = nx.cycle_basis(G)
        for ring_id, cycle in enumerate(rings):
            if len(cycle) < 3 or len(cycle) > max_ring:
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

        offset += node_count

    features = np.vstack(features_list)
    labels = np.vstack(labels_list)
    adjacency = sp.block_diag(adj_blocks, format='csr')
    edge_index = torch.tensor(np.array(all_edges).T, dtype=torch.long)

    v_cochain = Cochain(dim=0, x=torch.FloatTensor(features), y=torch.FloatTensor(labels))
    e_cochain = Cochain(dim=1, x=torch.ones((edge_index.shape[1], 1)), boundary_index=edge_index)

    if all_ring_rows:
        two_cell_boundary = torch.tensor([all_ring_rows, all_ring_cols], dtype=torch.long)
        two_cell_cochain = Cochain(dim=2, boundary_index=two_cell_boundary)
    else:
        two_cell_cochain = None

    complex_obj = Complex(v_cochain, e_cochain, two_cell_cochain)
    return features, adjacency, labels, complex_obj


def process_tu(data, class_num, max_ring=6):
    if isinstance(data, Batch):
        data_list = data.to_data_list()
    elif isinstance(data, Data):
        data_list = [data]
    elif isinstance(data, list):
        data_list = data
    elif hasattr(data, "__len__") and hasattr(data, "__getitem__") and not hasattr(data, "x"):
        data_list = [data[i] for i in range(len(data))]
    else:
        data_list = [data]

    ft_size = data_list[0].num_features
    num = range(class_num)
    labelnum = range(class_num, ft_size)

    features_list = []
    labels_list = []
    adj_blocks = []
    all_edges = []
    all_ring_rows = []
    all_ring_cols = []
    node_offset = 0
    edge_offset = 0
    ring_id = 0

    for graph in data_list:
        x = graph.x.cpu()
        edge_index = graph.edge_index.cpu()
        node_count = x.size(0)

        feature = x[:, num].numpy()
        label = x[:, labelnum].numpy()
        features_list.append(feature)
        labels_list.append(label)

        edges = [tuple(sorted(edge)) for edge in edge_index.t().tolist()]
        edges = sorted(set(edge for edge in edges if edge[0] != edge[1]))
        edge_to_id = {edge: idx for idx, edge in enumerate(edges)}

        if edges:
            rows, cols = zip(*edges)
            adj_blocks.append(sp.coo_matrix((np.ones(len(edges)), (rows, cols)), shape=(node_count, node_count)))
            all_edges.extend([[u + node_offset, v + node_offset] for u, v in edges])
        else:
            adj_blocks.append(sp.coo_matrix((node_count, node_count)))

        graph_nx = nx.Graph()
        graph_nx.add_nodes_from(range(node_count))
        graph_nx.add_edges_from(edges)
        for cycle in nx.cycle_basis(graph_nx):
            if len(cycle) < 3 or len(cycle) > max_ring:
                continue
            for i in range(len(cycle)):
                edge = tuple(sorted((cycle[i], cycle[(i + 1) % len(cycle)])))
                local_edge_id = edge_to_id.get(edge)
                if local_edge_id is not None:
                    all_ring_rows.append(edge_offset + local_edge_id)
                    all_ring_cols.append(ring_id)
            ring_id += 1

        node_offset += node_count
        edge_offset += len(edges)

    features = np.vstack(features_list)
    labels = np.vstack(labels_list)
    adjacency = sp.block_diag(adj_blocks, format='csr')

    if all_edges:
        edge_index = torch.tensor(np.array(all_edges).T, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    v_cochain = Cochain(dim=0, x=torch.FloatTensor(features), y=torch.FloatTensor(labels))
    e_cochain = Cochain(dim=1, x=torch.ones((edge_index.shape[1], 1)), boundary_index=edge_index)

    if all_ring_rows:
        two_cell_boundary = torch.tensor([all_ring_rows, all_ring_cols], dtype=torch.long)
        two_cell_cochain = Cochain(dim=2, boundary_index=two_cell_boundary, num_cells=ring_id)
    else:
        two_cell_cochain = Cochain(dim=2, boundary_index=torch.empty((2, 0), dtype=torch.long), num_cells=0)

    complex_obj = Complex(v_cochain, e_cochain, two_cell_cochain)
    return features, adjacency, labels, complex_obj

def micro_f1(logits, labels):
    # Compute predictions
    preds = torch.round(nn.Sigmoid()(logits))

    # Cast to avoid trouble
    preds = preds.long()
    labels = labels.long()

    # Count true positives, true negatives, false positives, false negatives
    tp = torch.nonzero(preds * labels).shape[0] * 1.0
    tn = torch.nonzero((preds - 1) * (labels - 1)).shape[0] * 1.0
    fp = torch.nonzero(preds * (labels - 1)).shape[0] * 1.0
    fn = torch.nonzero((preds - 1) * labels).shape[0] * 1.0

    # Compute micro-f1 score
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = (2 * prec * rec) / (prec + rec)
    return f1

"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""
def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data(dataset_str): # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    current_path = os.path.dirname(__file__)
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    return adj, features, labels, idx_train, idx_val, idx_test

def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



