import networkx as nx
from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='data', name='ENZYMES', use_node_attr=True)
data = dataset[2]
e_ind = data.edge_index.cpu().numpy().T

G = nx.Graph()
G.add_edges_from([tuple(sorted(e)) for e in e_ind.tolist()])
rings = nx.cycle_basis(G)
print(f"第一张图环数量：{len(rings)}")
