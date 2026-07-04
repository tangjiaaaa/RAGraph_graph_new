import torch
from .GCN import ExtendedGCN
from .gcnlayers import GcnLayers

def test_gcnlayers():
    # 假设有3个普通节点，2个胞腔节点，总5个节点
    num_nodes = 3
    total_nodes = 5
    feature_dim = 4
    hidden_dim = 8
    num_layers = 2
    dropout = 0.0

    # 随机生成特征，shape [1, total_nodes, feature_dim]，批次1个图
    features = torch.randn(1, total_nodes, feature_dim)

    # 构造扩展邻接矩阵 shape [total_nodes, total_nodes]
    adj = torch.zeros(total_nodes, total_nodes)
    # 普通节点间连边 (0-1,1-2)
    adj[0,1] = adj[1,0] = 1
    adj[1,2] = adj[2,1] = 1
    # 胞腔节点连边 (3-4)
    adj[3,4] = adj[4,3] = 1
    # 节点-胞腔连接 (0-3, 2-4)
    adj[0,3] = adj[3,0] = 1
    adj[2,4] = adj[4,2] = 1

    # 实例化改造后的 GcnLayers
    gcn_layers = GcnLayers(n_in=feature_dim, n_h=hidden_dim, num_layers_num=num_layers, dropout=dropout)

    # 前向计算
    output = gcn_layers(features, adj, sparse=False, num_nodes=num_nodes, LP=True)

    # 输出形状检查，应该是 [1, total_nodes, hidden_dim]
    print("Output shape:", output.shape)
    assert output.shape == (1, total_nodes, hidden_dim), "输出形状不正确"

    # 查看部分输出
    print("Output (first node):", output[0,0])

if __name__ == "__main__":
    test_gcnlayers()
