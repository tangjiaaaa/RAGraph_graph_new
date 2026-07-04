import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import scipy.sparse as sp

from preprompt import PrePrompt
from RAGraph import RAGraph  # 你的检索增强模型
from utils import process    # 你的 TU dataset 预处理

# === Dataset ===
dataset = TUDataset(root='data', name='COX2', use_node_attr=True)
feature_size = dataset.num_node_attributes
num_classes = dataset.num_classes

train_dataset = dataset[:int(0.5 * len(dataset))]
test_dataset = dataset[int(0.8 * len(dataset)):]

# === Load pretrained PrePrompt ===
pretrain_model = PrePrompt(feature_size, 256, 'prelu', 1, 0.3, use_proj=True).cuda()
pretrain_model.load_state_dict(torch.load('modelset/model_COX2.pkl'))
pretrain_model.eval()

# === Build RAGraph for NF ===
rag_model = RAGraph(
    pretrain_model,
    resource_dataset=train_dataset,
    feture_size=feature_size,
    num_class=num_classes,
    emb_size=256,
    finetune=False,
    noise_finetune=False
).cuda()

# === 直接写死 hopK 和 topK ===
rag_model.toy_graph_base.toy_graph_hop = 2  # 实际 hopK
rag_model.toy_graph_base.retrieve_num = 5   # 实际 topK

print("[DEBUG INIT] topK:", rag_model.toy_graph_base.retrieve_num)
print("[DEBUG INIT] hopK:", rag_model.toy_graph_base.toy_graph_hop)

rag_model.toy_graph_base.build_toy_graph(train_dataset)

# === Evaluate ===
rag_model.eval()
correct = 0
total = 0

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

for data in test_loader:
    features, adj, labels, ptr = process.process_tu(data, num_classes, feature_size)
    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

    if hasattr(adj, "todense"):
        adj = adj.todense()

    adj = torch.FloatTensor(adj)
    # === 保证全部是 Tensor ===
    if not isinstance(features, torch.Tensor):
        features = torch.FloatTensor(features)
    if not isinstance(adj, torch.Tensor):
        adj = torch.FloatTensor(adj)
    if not isinstance(labels, torch.Tensor):
        labels = torch.LongTensor(labels)

    features, adj, labels = features.cuda(), adj.cuda(), labels.cuda()

    logits = rag_model(features, adj, ptr=ptr)
    preds = logits.argmax(dim=1)

    correct += (preds == labels).sum().item()
    total += labels.size(0)

print(f"[NF] Test Accuracy: {100 * correct / total:.2f}%")
