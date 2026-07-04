import torch
import torch.nn as nn
from layers import GCN

class GcnLayers(torch.nn.Module):
    def __init__(self, n_in, n_h,num_layers_num,dropout):
        super(GcnLayers, self).__init__()

        self.act=torch.nn.ReLU()
        self.num_layers_num=num_layers_num
        self.g_net, self.bns = self.create_net(n_in,n_h,self.num_layers_num)
        self.out_dim = n_h
        self.dropout=torch.nn.Dropout(p=dropout)

    def create_net(self,input_dim, hidden_dim,num_layers):

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_layers):

            if i:
                nn = GCN(hidden_dim, hidden_dim)
            else:
                nn = GCN(input_dim, hidden_dim)
            conv = nn
            bn = torch.nn.BatchNorm1d(hidden_dim)

            self.convs.append(conv)
            self.bns.append(bn)

        return self.convs, self.bns

    def forward(self, seq, adj, sparse, LP=False, num_nodes_list=None):
        if seq.dim() == 3:
            seq = seq.squeeze(0)
        if adj.dim() == 3:
            adj = adj.squeeze(0)

        graph_output = seq
        xs = []
        for i in range(self.num_layers_num):
            input = (graph_output, adj)
            graph_output = self.convs[i](input)
            if LP:
                graph_output = self.bns[i](graph_output)
                graph_output = self.dropout(graph_output)
            xs.append(graph_output)

        return graph_output

    def inference(self, x, adj=None):
        """
        支持两种输入：
        - (features, adj) 结构图输入
        - Complex 对象输入（具有 nodes 和 edges）
        """
        if isinstance(x, tuple):  # (features, adj)
            return self.forward(*x, sparse=False, LP=False)

        elif hasattr(x, 'nodes') and hasattr(x, 'edges'):  # Complex 输入
            v_feat = x.nodes.x
            edge_index = x.edges.boundary_index

            num_nodes = v_feat.size(0)
            adj = torch.zeros((num_nodes, num_nodes), device=v_feat.device)
            adj[edge_index[0], edge_index[1]] = 1.0  # dense 邻接矩阵

            return self.forward(v_feat, adj, sparse=False, LP=False)

        else:
            raise ValueError(f"[inference] Unsupported input type: {type(x)}")

