import numpy as np
import scipy.sparse as sp

from preprompt import PrePrompt
import preprompt
from utils import process
import aug
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from ragraph_utils import seed_everything
def ring_contrastive_loss(features, ring_boundary, edge_index, max_rings=100):
    if ring_boundary.size(1) == 0:
        return torch.tensor(0.0, requires_grad=True).cuda()

    ring_dict = {}
    for i in range(ring_boundary.size(1)):
        edge_id = ring_boundary[0, i].item()
        ring_id = ring_boundary[1, i].item()
        u, v = edge_index[0, edge_id].item(), edge_index[1, edge_id].item()
        if ring_id not in ring_dict:
            ring_dict[ring_id] = set()
        ring_dict[ring_id].update([u, v])

    ring_embeddings = []
    for node_set in ring_dict.values():
        if len(node_set) < 2:
            continue
        node_feats = features[list(node_set)]
        ring_emb = node_feats.mean(dim=0)
        ring_embeddings.append(ring_emb)

    if len(ring_embeddings) < 2:
        return torch.tensor(0.0, requires_grad=True).cuda()

    ring_embeddings = torch.stack(ring_embeddings, dim=0)

    # 控制环数量，避免爆显存
    if ring_embeddings.size(0) > max_rings:
        idx = torch.randperm(ring_embeddings.size(0))[:max_rings]
        ring_embeddings = ring_embeddings[idx]

    # 构造简单的正对（相邻）
    anchor = ring_embeddings
    positive = torch.roll(ring_embeddings, shifts=1, dims=0)
    target = torch.ones(anchor.size(0)).to(anchor.device)

    loss = F.cosine_embedding_loss(anchor, positive, target, margin=0.3)
    return loss
parser = argparse.ArgumentParser("RAGraph")

parser.add_argument('--dataset', type=str, default="PROTEINS", help='data')
parser.add_argument('--aug_type', type=str, default="edge", help='aug type: mask or edge')
parser.add_argument('--drop_percent', type=float, default=0.1, help='drop percent')
parser.add_argument('--seed', type=int, default=39, help='seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
args = parser.parse_args()
args.save_name = f'modelset/model_{args.dataset}.pkl'

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device("cuda")

print('-' * 100)
print(args)
print('-' * 100)

seed_everything(args.seed)

from torch_geometric.datasets import TUDataset
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
# 自定义 Collate 函数
class Collater:
    def __init__(self, follow_batch=None):
        # 如果 follow_batch 是 None，则使用空列表
        if follow_batch is None:
            follow_batch = []
        self.follow_batch = follow_batch

    def collate(self, batch):
        # 使用 Batch.from_data_list 来将多个 Data 对象合并成一个批次
        return Batch.from_data_list(batch, self.follow_batch)

    def __call__(self, batch):
        return self.collate(batch)
sparse = False  # Define sparse before usage

dataset = TUDataset(root='data', name=args.dataset, use_node_attr=True)
collater = Collater()
loader = DataLoader(dataset, batch_size=(4 if args.dataset == 'ENZYMES' else 4), shuffle=True, drop_last=True, collate_fn=collater)
ft_size = dataset.num_node_attributes

model = PrePrompt(ft_size, 256, 'prelu', 1, 0.3).to(device)

best = 1e9
patience = 10
cnt_wait = 0


for epoch in range(50):
    seed_everything(args.seed)
    total_loss = 0
    print(f"Epoch {epoch} start")
    for step, data in enumerate(loader):
        print(f"Step {step} start")
        features, adj, nodelabels , complex_obj = process.process_tu(data, ft_size)
        print("features.shape:", features.shape)
        print("adj type:", type(adj))
        if hasattr(adj, 'shape'):
            print("adj.shape:", adj.shape)
        else:
            print("adj has no 'shape' attribute, type is:", type(adj))
        negative_sample = preprompt.prompt_pretrain_sample(adj, 100)

        nb_nodes = features.shape[0]
        num_nodes_list = [nb_nodes]

        features = torch.FloatTensor(features[np.newaxis]).to(device)
        aug_features1edge = features.clone()
        aug_features2edge = features.clone()

        aug_adj1edge = aug.aug_random_edge(adj, drop_percent=args.drop_percent)
        aug_adj2edge = aug.aug_random_edge(adj, drop_percent=args.drop_percent)

        adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
        adj = torch.FloatTensor(np.asarray(adj.todense())[np.newaxis]).to(device)

        aug_adj1edge = process.normalize_adj(aug_adj1edge + sp.eye(aug_adj1edge.shape[0]))
        aug_adj1edge = torch.FloatTensor(np.asarray(aug_adj1edge.todense())[np.newaxis]).to(device)

        aug_adj2edge = process.normalize_adj(aug_adj2edge + sp.eye(aug_adj2edge.shape[0]))
        aug_adj2edge = torch.FloatTensor(np.asarray(aug_adj2edge.todense())[np.newaxis]).to(device)

        labels = torch.FloatTensor(nodelabels[np.newaxis]).to(device)

        optimiser = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
        model.train()
        optimiser.zero_grad()

        idx = np.random.permutation(nb_nodes)
        shuf_fts = features[:, idx, :].to(device)

        lbl = torch.cat((torch.ones(1, nb_nodes), torch.zeros(1, nb_nodes)), 1).to(device)

        logit = model(features, shuf_fts, aug_features1edge, aug_features2edge,
                      adj, aug_adj1edge, aug_adj2edge, sparse,
                      None, None, None, lbl=lbl, sample=negative_sample, num_nodes_list=num_nodes_list)

        # ==== 添加基于环结构的 contrastive loss ====
        if complex_obj.cochains[2] is not None:
            ring_boundary = complex_obj.cochains[2].boundary_index.cuda()
            edge_index = complex_obj.cochains[1].boundary_index.cuda()
            ring_loss = ring_contrastive_loss(features.squeeze(0), ring_boundary, edge_index)
            print(f"[Debug] ring_loss: {ring_loss.item():.4f} | main_loss: {logit.item():.4f}")

            logit += ring_loss * 0.1

        total_loss += logit.item()
        logit.backward()
        optimiser.step()

        print(f"Step {step} loss: {logit.item():.4f}")

    avg_loss = total_loss / (step + 1)
    print(f"Epoch {epoch} average loss: {avg_loss:.4f}")

    if avg_loss < best:
        best = avg_loss
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), args.save_name)
    else:
        cnt_wait += 1
    if cnt_wait == patience:
        print('Early stopping!')
        break
    print(f"End of epoch {epoch}, best epoch so far: {best_t}")
