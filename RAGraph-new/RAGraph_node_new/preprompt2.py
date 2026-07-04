import torch
import torch.nn as nn
import torch.nn.functional as F
from models import DGI, GraphCL, Lp, GcnLayers
from layers import AvgReadout
import numpy as np


def get_subgraph_3(feature, adj):
    # 计算三跳邻接（保持二跳也是类似）
    adj_3hop = torch.matmul(adj, torch.matmul(adj, adj)).squeeze()
    adj_3hop[adj_3hop > 0] = 1

    index = torch.nonzero(adj_3hop, as_tuple=False)

    res = torch.zeros(feature.size(0), feature.size(1), device=feature.device)
    cnt = torch.zeros(feature.size(0), device=feature.device)

    # 累加三跳节点特征
    for src, dst in index:
        res[src] += feature[dst]
        cnt[src] += 1

    # 避免除以 0
    cnt_safe = cnt.clone()
    cnt_safe[cnt_safe == 0] = 1

    # 批量除法
    res = res / cnt_safe.unsqueeze(1)

    return res



class PrePrompt(nn.Module):
    def __init__(self, n_in, n_h, activation, num_layers_num, p):
        super(PrePrompt, self).__init__()
        self.dgi = DGI(n_h)
        self.graphcledge = GraphCL(n_in, n_h, activation)
        self.graphclmask = GraphCL(n_in, n_h, activation)
        self.lp = Lp(n_in, n_h)
        self.gcn = GcnLayers(n_in, n_h, num_layers_num, p)
        self.read = AvgReadout()

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, seq1, seq2, seq3, seq4, adj, aug_adj1edge, aug_adj2edge,
                sparse, msk, samp_bias1, samp_bias2,
                lbl, sample,num_nodes_list):
        negative_sample = torch.tensor(sample, dtype=int).cuda()
        seq1 = torch.squeeze(seq1, 0)
        seq2 = torch.squeeze(seq2, 0)
        seq3 = torch.squeeze(seq3, 0)
        seq4 = torch.squeeze(seq4, 0)
        logits3 = self.lp(self.gcn, seq1, adj, sparse, num_nodes_list=num_nodes_list)
        lploss = compareloss(logits3, negative_sample, temperature=1.5)
        lploss.requires_grad_(True)

        ret = lploss
        return ret

    def embed(self, seq, adj, sparse, msk, LP):
        h_1 = self.gcn(seq, adj, sparse, LP)
        h = h_1.squeeze()
        sub_3_feature = get_subgraph_3(h, adj)
        c = self.read(sub_3_feature, msk)
        return h.detach(), c.detach()

    # def inference(self, features, adj):
    def inference(self, *args):
        # 如果是普通图输入 (features, adj)
        if len(args) == 2:  # 确保是两个输入
            features, adj = args  # 将 args 中的两个元素解包成 features 和 adj
            h, _ = self.embed(features, adj, False, None, False)
            return h
            # 如果是 Complex 图输入
        elif len(args) == 1 and hasattr(args[0], "is_complex") and args[0].is_complex:
            return self._inference_complex(args[0])  # 调用处理 Complex 的方法

        else:
            raise ValueError("Unsupported input type for inference")

    def _inference_complex(self, complex_obj):
        """
            处理 Complex 图对象的嵌入逻辑
            complex_obj: Complex 实例，包含高阶结构信息，如 Cochain、boundary_index 等
            """
        try:
            # 获取节点特征（0维 Cochain 的特征）
            x = complex_obj.nodes.x  # 0维节点特征
            # 获取边连接关系（1维 Cochain 的边连接关系）
            edge_index = complex_obj.edges.boundary_index  # 1维边连接关系
            # 获取环的边界信息（2维 Cochain）
            if complex_obj.two_cells is not None:
               two_cell_boundary_index = complex_obj.two_cells.boundary_index  # 环（2维）的边界
            else:
               two_cell_boundary_index = None
            num_nodes = x.size(0)  # 获取节点数
            adj = torch.zeros((num_nodes, num_nodes), device=x.device)  # 创建邻接矩阵
            adj[edge_index[0], edge_index[1]] = 1  # 用边连接关系填充邻接矩阵
            # 用 GCN 处理 Complex 图
            num_nodes_list = [num_nodes]  # 节点数量列表
            h_1 = self.gcn(x, adj, sparse=False, num_nodes_list=num_nodes_list, LP=False)
            # 处理环（2维 Cochain）
            if two_cell_boundary_index is not None:
                # 这里可以加入额外的处理环的逻辑
                # 比如，你可以聚合环上的特征，或者计算环的嵌入
                ring_embedding = self.process_ring(two_cell_boundary_index, x)  # 处理环的边界特征
                h_1 = h_1 + ring_embedding  # 将环的嵌入信息加到最终的节点嵌入中

            return h_1.squeeze().detach()  # 返回图嵌入（去除维度，detach 防止反向传播）
        except Exception as e:
            print("[Complex inference error]:", e)
            return torch.zeros((1, self.gcn.out_dim), device=x.device)  # 出错时返回零向量

    def process_ring(self, ring_boundary_index, node_features):
        """
        处理环（2维单元）的特征聚合。
        ring_boundary_index: 环的边界信息
        node_features: 节点特征
        """
        # 你可以在这里添加环的特征聚合逻辑，例如：
        # - 聚合环的特征
        # - 将环的特征与节点特征结合
        ring_embedding = torch.zeros_like(node_features)

        # 这里的例子是简单的特征聚合
        for i in range(ring_boundary_index.size(1)):
            ring_embedding += node_features[ring_boundary_index[0, i]]  # 简单的聚合：环边界上的节点特征加总

        # 返回环的嵌入信息（可以加上加权的逻辑）
        return ring_embedding







def mygather(feature, index):
    input_size = index.size(0)
    index = index.flatten()
    index = index.reshape(len(index), 1)
    index = torch.broadcast_to(index, (len(index), feature.size(1)))

    res = torch.gather(feature, dim=0, index=index)
    # print("res", res.shape)
    return res.reshape(input_size, -1, feature.size(1))


def compareloss(feature, tuples, temperature):
    h_tuples = mygather(feature, tuples)  # negative
    # print("tuples",h_tuples.shape)
    temp = torch.arange(0, len(tuples))
    temp = temp.reshape(-1, 1)
    temp = torch.broadcast_to(temp, (temp.size(0), tuples.size(1)))
    temp = temp.cuda()
    h_i = mygather(feature, temp)  # positive

    sim = F.cosine_similarity(h_i, h_tuples, dim=2)
    # print("sim",sim)
    exp = torch.exp(sim)
    exp = exp / temperature
    exp = exp.permute(1, 0)
    numerator = exp[0].reshape(-1, 1)
    denominator = exp[1:exp.size(0)]
    denominator = denominator.permute(1, 0)
    denominator = denominator.sum(dim=1, keepdim=True)

    # print("numerator",numerator)
    # print("denominator",denominator)
    res = -1 * torch.log(numerator / denominator)
    return res.mean()


def prompt_pretrain_sample(adj, n):
    # print("adj.shape", adj.shape)
    nodenum = adj.shape[0]
    n = min(n, nodenum)
    indices = adj.indices
    indptr = adj.indptr
    res = np.zeros((nodenum, 1 + n))
    whole = np.array(range(nodenum))
    # print("#############")
    # print("start sampling disconnected tuples")
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


