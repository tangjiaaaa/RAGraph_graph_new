import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models import DGI, GraphCL, Lp, GcnLayers
from layers import AvgReadout


class PrePrompt(nn.Module):
    def __init__(self, n_in, n_h, activation, num_layers_num, p, use_proj=False, proj_dim=256):
        super(PrePrompt, self).__init__()
        self.gcn = GcnLayers(n_in, n_h, num_layers_num, p)
        self.lp = Lp(n_in, n_h)
        self.read = AvgReadout()

        self.use_proj = use_proj
        self.out_dim = n_h
        if self.use_proj:
            self.out_proj = nn.Linear(n_h, proj_dim)
            self.out_dim = proj_dim

    def forward(self, seq1, seq2, seq3, seq4, adj, aug_adj1edge, aug_adj2edge,
                sparse, msk, samp_bias1, samp_bias2,
                lbl, sample):
        """
        如果用 LP 任务，这里示例只保留对齐的输入
        如果用 LP 任务，这里示例只保留对齐的输入
        """
        negative_sample = torch.tensor(sample, dtype=int).cuda()
        seq1 = torch.squeeze(seq1, 0)
        logits3 = self.lp(self.gcn, seq1, adj, sparse)
        lploss = compareloss(logits3, negative_sample, temperature=1.5)
        lploss.requires_grad_(True)
        return lploss

    def embed(self, seq, adj, sparse=False, msk=None, LP=False):
        """
        关键：
          - 输入是拼接好的全局 block-diag
          - 不做 mask 和 pooling，保留所有节点嵌入
        """
        # print(f"[embed] input seq: {seq.shape}, adj: {adj.shape}")
        h_1 = self.gcn(seq, adj, sparse, LP)
        # print(f" [embed] output node_emb: {h_1.shape}")
        return h_1, None

    def inference(self, *args):
        """
        支持单张图或单个 Complex 的推理：
        - 返回节点级别嵌入
        - 投影后维度和 resource_keys 对齐
        - 不在这里做 graph-level mean pooling
        """
        if len(args) == 2:
            features, adj = args
            node_emb, _ = self.embed(features, adj, sparse=False)
            if self.use_proj:
                node_emb = self.out_proj(node_emb)
            return node_emb.detach()  # shape: [N, D]

        elif len(args) == 1 and hasattr(args[0], "is_complex") and args[0].is_complex:
            return self._inference_complex(args[0])

        else:
            raise ValueError("Unsupported input type for inference")

    def _inference_complex(self, complex_obj):
        """
        单图 Complex 推理，局部索引，照旧
        """
        try:
            x = complex_obj.nodes.x
            edge_index = complex_obj.edges.boundary_index
            two_cell_boundary_index = getattr(complex_obj.two_cells, "boundary_index", None)

            num_nodes = x.size(0)
            adj = torch.zeros((num_nodes, num_nodes), device=x.device)
            adj[edge_index[0], edge_index[1]] = 1

            h_1 = self.gcn(x, adj, sparse=False, num_nodes_list=[num_nodes], LP=False)

            if two_cell_boundary_index is not None:
                ring_embedding = self.process_ring(two_cell_boundary_index, x)
                h_1 = h_1 + ring_embedding

            h_1 = h_1.mean(dim=0)
            if self.use_proj:
                h_1 = self.out_proj(h_1)

            return h_1.detach()

        except Exception as e:
            print("[Complex inference error]:", e)
            return torch.zeros((self.out_dim,), device=x.device)

    def process_ring(self, ring_boundary_index, edge_index, node_emb):
        """
        完整点 → 边 → 环 coboundary 路径：
          - 先生成边 embedding: 聚合两端点
          - 再对属于同一环的边做 mean
        """
        device = node_emb.device

        # 1) 边层：先生成所有边 embedding
        edge_emb = torch.zeros((edge_index.size(1), node_emb.size(1)), device=device)
        for e in range(edge_index.size(1)):
            u = edge_index[0, e].item()
            v = edge_index[1, e].item()
            edge_emb[e] = (node_emb[u] + node_emb[v]) / 2.0  # 也可 concat 或 mlp

        # 2) 环层：对每个环聚合它的边 embedding
        ring_dict = {}  # ring_id : set of edge indices
        for i in range(ring_boundary_index.size(1)):
            edge_id = ring_boundary_index[0, i].item()
            ring_id = ring_boundary_index[1, i].item()
            ring_dict.setdefault(ring_id, []).append(edge_id)

        ring_embeddings = []
        for edge_ids in ring_dict.values():
            edges = edge_emb[edge_ids]  # shape [num_edges, emb_dim]
            ring_emb = edges.mean(dim=0)  # 该环的 embedding
            ring_embeddings.append(ring_emb)

        if ring_embeddings:
            ring_embeddings = torch.stack(ring_embeddings)
        else:
            ring_embeddings = torch.zeros((1, node_emb.size(1)), device=device)

        return ring_embeddings  # shape [num_rings, emb_dim]


def mygather(feature, index):
    input_size = index.size(0)
    index = index.flatten().reshape(-1, 1)
    index = torch.broadcast_to(index, (len(index), feature.size(1)))
    res = torch.gather(feature, dim=0, index=index)
    return res.reshape(input_size, -1, feature.size(1))


def compareloss(feature, tuples, temperature):
    """
    对比学习损失，和拼接后的输入一致
    """
    h_tuples = mygather(feature, tuples)
    temp = torch.arange(0, len(tuples)).reshape(-1, 1)
    temp = torch.broadcast_to(temp, (temp.size(0), tuples.size(1)))
    temp = temp.cuda()
    h_i = mygather(feature, temp)
    sim = F.cosine_similarity(h_i, h_tuples, dim=2)
    exp = torch.exp(sim) / temperature
    exp = exp.permute(1, 0)
    numerator = exp[0].reshape(-1, 1)
    denominator = exp[1:].permute(1, 0).sum(dim=1, keepdim=True)
    res = -1 * torch.log(numerator / denominator)
    return res.mean()


def prompt_pretrain_sample(adj, n):
    """
    在拼接后邻接矩阵上采样负样本
    """
    nodenum = adj.shape[0]
    indices = adj.indices
    indptr = adj.indptr
    res = np.zeros((nodenum, 1 + n))
    whole = np.array(range(nodenum))
    for i in range(nodenum):
        nonzero_index_i_row = indices[indptr[i]:indptr[i + 1]]
        zero_index_i_row = np.setdiff1d(whole, nonzero_index_i_row)
        np.random.shuffle(nonzero_index_i_row)
        np.random.shuffle(zero_index_i_row)
        if np.size(nonzero_index_i_row) == 0:
            res[i][0] = i
        else:
            res[i][0] = nonzero_index_i_row[0]
        res[i][1:1 + n] = zero_index_i_row[0:n]
    return res.astype(int)
