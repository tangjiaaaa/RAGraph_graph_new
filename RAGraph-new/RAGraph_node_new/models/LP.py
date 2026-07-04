# 这段代码是位于 ragraph_node/models/LP.py 的 Lp 模块，
# 功能相对简单，结合了一个 GCN 编码器和 ELU 激活函数，
# 并且定义了一个可学习的提示向量（prompt），
# 但目前该提示向量在前向传播中没有直接被使用。
# 它很可能是在 RAGRAPH 框架中用于某种特定任务或辅助模块，比如作为一种表示或特征变换器。
import torch
import torch.nn as nn

class Lp(nn.Module):
    def __init__(self, n_in, n_h):
        super(Lp, self).__init__()
        self.sigm = nn.ELU()
        self.prompt = nn.Parameter(torch.FloatTensor(1, n_h), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.prompt)

    def forward(self, gcn, seq, adj, sparse, num_nodes_list):
        """
        gcn: GcnLayers 模块实例
        seq: [B, N+M, d] 节点特征
        adj: [B, N+M, N+M] 邻接矩阵
        sparse: 是否为稀疏邻接
        num_nodes_list: [B] 每个图的普通节点数列表

        返回:
        [B, N+M, hidden_dim] -> 经激活后输出
        """
        h_1 = gcn(seq, adj, sparse, num_nodes_list)
        ret = self.sigm(h_1.squeeze(dim=0))  # 若只有 1 个图，squeeze
        return ret


