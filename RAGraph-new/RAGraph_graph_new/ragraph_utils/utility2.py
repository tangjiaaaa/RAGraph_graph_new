import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data, Batch
import random, os
import networkx as nx
from .complex import Cochain, Complex, ComplexBatch
from .extract_ring import ring_contrastive_loss
from collections import deque


def seed_everything(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def normalize_adj(adj):
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def find_cycles_using_spanning_tree(G, max_k=6):
    def find(parent, i):
        if parent[i] != i:
            parent[i] = find(parent, parent[i])
        return parent[i]

    def union(parent, rank, x, y):
        root_x = find(parent, x)
        root_y = find(parent, y)
        if root_x == root_y:
            return
        if rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
        elif rank[root_x] > rank[root_y]:
            parent[root_y] = root_x
        else:
            parent[root_y] = root_x
            rank[root_x] += 1

    vertices = list(G.nodes())
    edges = [(u, v) for u, v in G.edges() if u < v]
    parent = {v: v for v in vertices}
    rank = {v: 0 for v in vertices}
    tree_edges, non_tree_edges = [], []

    for u, v in edges:
        if find(parent, u) != find(parent, v):
            tree_edges.append((u, v))
            union(parent, rank, u, v)
        else:
            non_tree_edges.append((u, v))

    tree = {v: [] for v in vertices}
    for u, v in tree_edges:
        tree[u].append(v)
        tree[v].append(u)

    cycles = []
    for u, v in non_tree_edges:
        visited = {v_: False for v_ in vertices}
        parent_map = {v_: None for v_ in vertices}
        queue = [u]
        visited[u] = True
        while queue:
            curr = queue.pop(0)
            if curr == v:
                break
            for nbr in tree[curr]:
                if not visited[nbr]:
                    visited[nbr] = True
                    parent_map[nbr] = curr
                    queue.append(nbr)
        path = []
        cur = v
        while cur is not None:
            path.append(cur)
            cur = parent_map[cur]
        if 3 <= len(path) <= max_k:
            cycles.append(path)
    return cycles


def extract_fcb_rings(edges_list, max_ring=20, method="fcb"):
    edges_list = sorted(list(set([tuple(sorted(e)) for e in edges_list])))
    edge2id = {e: i for i, e in enumerate(edges_list)}
    G = nx.Graph()
    G.add_edges_from(edges_list)
    if method == "fcb":
        rings = find_cycles_using_spanning_tree(G, max_k=max_ring)
    elif method == "nx":
        rings = nx.cycle_basis(G)
    else:
        raise NotImplementedError(f"Unknown method: {method}")

    ring_rows, ring_cols = [], []
    for idx, cycle in enumerate(rings):
        if len(cycle) < 3 or len(cycle) > max_ring:
            continue
        for j in range(len(cycle)):
            u, v = cycle[j], cycle[(j + 1) % len(cycle)]
            edge = tuple(sorted([u, v]))
            if edge in edge2id:
                ring_rows.append(edge2id[edge])
                ring_cols.append(idx)
            else:
                print(f"[WARN] edge {edge} not found in edge list")

    return edges_list, ring_rows, ring_cols


def process_tu_dataset(data, num_classes, node_attr_dim, max_ring=10):
    def build_single_graph(x, edge_index, label):
        edge_index = torch.cat([edge_index, edge_index[[1, 0], :]], dim=1)
        edges = [tuple(sorted(e)) for e in edge_index.T.tolist()]
        edges_list, ring_rows, ring_cols = extract_fcb_rings(edges, max_ring=max_ring, method="fcb")

        # --- 2-Cell (Ring) ---
        if ring_rows:
            two_cell_boundary = torch.tensor([ring_rows, ring_cols], dtype=torch.long)
        else:
            # 使用包含有效形状的占位张量，防止混合批处理时底层复形库切片错位
            two_cell_boundary = torch.tensor([[0], [0]], dtype=torch.long)

        two_cell_cochain = Cochain(dim=2, boundary_index=two_cell_boundary)

        # --- 0-Cell (Node) ---
        v_cochain = Cochain(dim=0, x=torch.FloatTensor(x))

        # --- 1-Cell (Edge) 边界修复 ---
        b1_rows, b1_cols = [], []
        for edge_id, (u, v) in enumerate(edges_list):
            b1_rows.extend([u, v])
            b1_cols.extend([edge_id, edge_id])

        if len(b1_rows) > 0:
            one_cell_boundary = torch.tensor([b1_rows, b1_cols], dtype=torch.long)
        else:
            one_cell_boundary = torch.tensor([[0, 0], [0, 0]], dtype=torch.long)
            if len(edges_list) == 0:
                edges_list = [(0, 0)]

        e_cochain = Cochain(dim=1, x=torch.ones(len(edges_list), 1), boundary_index=one_cell_boundary)

        complex_obj = Complex(v_cochain, e_cochain, two_cell_cochain)

        # --- 邻接矩阵 (Adjacency Matrix) 修复 ---
        row = [e[0] for e in edges_list] + [e[1] for e in edges_list]
        col = [e[1] for e in edges_list] + [e[0] for e in edges_list]
        data_ones = np.ones(len(row))
        adj = sp.coo_matrix((data_ones, (row, col)), shape=(x.shape[0], x.shape[0]))

        return complex_obj, x, adj, label

    if isinstance(data, Batch):
        features_list, adj_list, labels, complex_list, ptr_list = [], [], [], [], [0]
        for i in range(data.num_graphs):
            node_mask = (data.batch == i)
            x = data.x[node_mask, :node_attr_dim].cpu().numpy()
            e_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
            e_ind = data.edge_index[:, e_mask].cpu().numpy()
            old2new = {old: new for new, old in enumerate(torch.where(node_mask)[0].tolist())}
            e_ind = np.vectorize(old2new.get)(e_ind)
            edge_index = torch.tensor(e_ind, dtype=torch.long)

            complex_obj, feat, adj, label = build_single_graph(x, edge_index, data.y[i].item())
            features_list.append(feat)
            adj_list.append(adj)
            labels.append(label)
            complex_list.append(complex_obj)
            ptr_list.append(ptr_list[-1] + feat.shape[0])

        features = np.vstack(features_list)
        adj = sp.block_diag(adj_list)
        graph_labels = torch.tensor(labels).long().cuda()
        ptr = torch.tensor(ptr_list).long().cuda()
        batch = torch.cat(
            [torch.full((feat.shape[0],), i, dtype=torch.long) for i, feat in enumerate(features_list)]).cuda()

    elif isinstance(data, Data):
        x = data.x[:, :node_attr_dim].cpu().numpy()
        edge_index = data.edge_index.cpu()
        complex_obj, features, adj, label = build_single_graph(x, edge_index, data.y.item())
        complex_list = [complex_obj]
        graph_labels = torch.tensor([label]).long().cuda()
        ptr = None
        batch = torch.zeros(features.shape[0], dtype=torch.long).cuda()

    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    complex_batch = ComplexBatch.from_complex_list(complex_list, max_dim=2)
    adj = normalize_adj(sp.csr_matrix(adj) + sp.eye(adj.shape[0])).todense()
    features = torch.FloatTensor(features).cuda()
    adj = torch.FloatTensor(adj).cuda()

    return features, adj, graph_labels, complex_batch, batch
