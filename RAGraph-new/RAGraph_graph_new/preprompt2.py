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
        self.ring_mask_weight = nn.Parameter(torch.ones(1))  # 可以初始化为1或0.5

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
        node_emb = self.gcn(seq1, adj, sparse)
        # 条件注入环上下文
        if hasattr(self, 'complex_obj'):
            print("Forward: type self.complex_obj", type(self.complex_obj))
            node_emb = self.prompt_injection(node_emb)

        logits3 = self.lp(lambda *_: node_emb, seq1, adj, sparse)  # 这里让 LP 用更新后的 emb
        # logits3 = self.lp(self.gcn, seq1, adj, sparse)
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
        return h_1.detach(), None

    def inference(self, *args):
        """
        普通 inference 不动，单张图可正常跑
        """
        if len(args) == 2:
            features, adj = args
            h, _ = self.embed(features, adj, False, None, False)
            # h = h.mean(dim=0) if h.dim() > 1 else h
            if self.use_proj:
                h = self.out_proj(h)
            return h.detach()

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
        正确的环聚合：先找到 edge → node，再聚合 node
        """
        ring_embedding = torch.zeros_like(node_emb).to(node_emb.device)
        for i in range(ring_boundary_index.size(1)):
            edge_id = ring_boundary_index[0, i].item()
            if edge_id >= edge_index.size(1):
                continue

            u = edge_index[0, edge_id].item()
            v = edge_index[1, edge_id].item()

            if u >= node_emb.size(0) or v >= node_emb.size(0):
                continue

            ring_embedding[u] += node_emb[u]
            ring_embedding[v] += node_emb[v]

        return ring_embedding

    def prompt_injection(self,  base_embedding):
        """
        在 Prompt 阶段决定是否注入环上下文
        base_context: Query Graph 的 embedding + 检索到的邻居/边上下文
        complex_obj: 检索到的 toy graph Complex
        """
        complex_obj = self.complex_obj  #
        two_cell_boundary_index = getattr(complex_obj.two_cells, "boundary_index", None)
        if two_cell_boundary_index is not None:
            print("[Prompt Mask] Has ring:", complex_obj.two_cells is not None)

            ring_embedding = self.process_ring(two_cell_boundary_index, complex_obj.edges.boundary_index,base_embedding)
            ring_embedding = ring_embedding.to(base_embedding.device)

            # 加权 Mask
            alpha = torch.sigmoid(self.ring_mask_weight)  # 可学习 gating
            base_embedding = base_embedding + alpha * ring_embedding.mean(dim=0)
        else:
            # 没有环，什么都不做就是 fallback
            pass
        return base_embedding


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
