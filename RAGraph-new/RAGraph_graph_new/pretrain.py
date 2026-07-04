import os
import argparse
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import global_add_pool
from preprompt import PrePrompt
import preprompt
from utils import process
import aug

# === Argument ===
parser = argparse.ArgumentParser("Pretrain with Ring Loss")
parser.add_argument('--generate_fewshot', action='store_true',
                    help='Generate few-shot support sets after pretraining')
parser.add_argument('--shotnum', type=int, default=5, help='K-shot')
parser.add_argument('--test_times', type=int, default=100, help='How many few-shot tasks to sample')
parser.add_argument('--dataset', type=str, default="ENZYMES")
#PROTEINS   ENZYMES
parser.add_argument('--drop_percent', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()
args.save_name = f'modelset/model_{args.dataset}.pkl'

print('-'*100)
print(args)
print('-'*100)

# === Seed & Env ===
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# === Dataset ===
dataset = TUDataset(root='data', name=args.dataset, use_node_attr=True)
loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

ft_size = dataset.num_node_attributes
class_num = dataset.num_classes

# === Model ===
model = PrePrompt(ft_size, 256, 'prelu', 1, 0.3, use_proj=True, proj_dim=256).cuda()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

best = 1e9
patience = 100
cnt_wait = 0
def nt_xent(anchor, positive, temperature=0.5):
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    logits = torch.mm(anchor, positive.T) / temperature
    labels = torch.arange(anchor.size(0)).cuda()
    return F.cross_entropy(logits, labels)
# === Ring Loss ===
def ring_contrastive_loss(node_emb, ring_boundary, edge_index):
    if ring_boundary.size(1) == 0:
        print(" ring_boundary empty")
        return torch.tensor(0.0, requires_grad=True).cuda()

    skipped_edgeid, skipped_nodeid = 0, 0
    ring_dict = {}

    for i in range(ring_boundary.size(1)):
        edge_id = ring_boundary[0, i].item()
        if edge_id >= edge_index.size(1):
            skipped_edgeid += 1
            continue
        u = edge_index[0, edge_id].item()
        v = edge_index[1, edge_id].item()
        if u >= node_emb.size(0) or v >= node_emb.size(0):
            skipped_nodeid += 1
            continue
        ring_id = ring_boundary[1, i].item()
        ring_dict.setdefault(ring_id, set()).update([u, v])

    # print(f"跳过 edge_id 越界: {skipped_edgeid}")
    # print(f"跳过 node_id 越界: {skipped_nodeid}")

    ring_embeddings = []
    for node_set in ring_dict.values():
        if len(node_set) < 2:
            continue
        ring_emb = node_emb[list(node_set)].mean(dim=0)
        ring_embeddings.append(ring_emb)

    if len(ring_embeddings) < 2:
        print(f"有效环数量不足，最终 ring_dict 有效: {len(ring_dict)}")
        return torch.tensor(0.0, requires_grad=True).cuda(), torch.zeros(node_emb.size(1), requires_grad=True).cuda()

    ring_embeddings = torch.stack(ring_embeddings)
    anchor = ring_embeddings
    positive = torch.roll(ring_embeddings, shifts=1, dims=0)
    target = torch.ones(anchor.size(0)).to(anchor.device)
    # with torch.no_grad():
    #     sims = F.cosine_similarity(anchor, positive)
    #     print(f"[ring_loss] 环对相似度分布: mean={sims.mean():.4f}, min={sims.min():.4f}, max={sims.max():.4f}")
    loss = F.cosine_embedding_loss(anchor, positive, target, margin=0.3)

    ring_mean = ring_embeddings.mean(dim=0)  # 聚合后的局部环特征
    # print(f" node_emb: {node_emb.shape}, edge_index: {edge_index.shape}, 有效环: {len(ring_dict)}")
    return loss, ring_mean

# = Train ===
for epoch in range(1000):
    model.train()
    total_loss = 0
    total_main_loss = 0
    total_ring_loss = 0
    for step, data in enumerate(loader):
        features, adj, labels, complex_obj, batch = process.process_tu(data, class_num, ft_size)
        # print(f" 拼接后 features.shape[0]: {features.shape[0]}")
        # print(f" 拼接后 edge_index max node id: {complex_obj.cochains[1].boundary_index.max().item()}")
        # === Debug 检查拼接是否一致 ===
        # if complex_obj.cochains[2] is not None:
        #     ring_boundary = complex_obj.cochains[2].boundary_index
        #     edge_index = complex_obj.cochains[1].boundary_index
            # print(f"ring_boundary: max edge_id {ring_boundary[0].max().item()} vs edge_index.size(1): {edge_index.size(1)}")

        negative_sample = preprompt.prompt_pretrain_sample(adj, 50)

        # === Feature & Adj ===
        features = torch.FloatTensor(features)
        adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
        if hasattr(adj, "todense"): adj = adj.todense()
        adj = torch.FloatTensor(adj)
        features, adj = features.cuda(), adj.cuda()

        # === Augment ===
        aug_features1 = aug_features2 = features
        aug_adj1 = aug.aug_random_edge(adj.cpu().numpy(), drop_percent=args.drop_percent)
        aug_adj1 = process.normalize_adj(aug_adj1 + sp.eye(aug_adj1.shape[0]))
        if hasattr(aug_adj1, "todense"): aug_adj1 = aug_adj1.todense()
        aug_adj1 = torch.FloatTensor(aug_adj1).cuda()

        aug_adj2 = aug.aug_random_edge(adj.cpu().numpy(), drop_percent=args.drop_percent)
        aug_adj2 = process.normalize_adj(aug_adj2 + sp.eye(aug_adj2.shape[0]))
        if hasattr(aug_adj2, "todense"): aug_adj2 = aug_adj2.todense()
        aug_adj2 = torch.FloatTensor(aug_adj2).cuda()

        optimiser.zero_grad()

        # === Ring loss ===
        # === 有环 ===
        if complex_obj.cochains[2] is not None:
            ring_boundary = complex_obj.cochains[2].boundary_index.cuda()
            edge_index = complex_obj.cochains[1].boundary_index.cuda()

            node_emb, _ = model.embed(features, adj, sparse=False, msk=None, LP=False)
            ring_loss, ring_mean = ring_contrastive_loss(node_emb, ring_boundary, edge_index)



            graph_emb = global_add_pool(node_emb, batch.cuda())  # [B,D]
            graph_emb = F.normalize(graph_emb, dim=-1)  # 保持范围稳定
            graph_emb = graph_emb + torch.randn_like(graph_emb) * 0.05  # 加强扰动

        else:
            node_emb, _ = model.embed(features, adj, sparse=False, msk=None, LP=False)
            ring_loss = torch.tensor(0.0).cuda()

            graph_emb = global_add_pool(node_emb, batch.cuda())  # 必须和上面一致
            graph_emb = F.normalize(graph_emb, dim=-1)
            graph_emb = graph_emb + torch.randn_like(graph_emb) * 0.05

            ring_mean = torch.zeros(node_emb.size(1)).cuda()  # shape [D]
            ring_mean = ring_mean + torch.randn_like(ring_mean) * 0.1
        # === 拼接 ===
        ring_mean = ring_mean.unsqueeze(0).expand(graph_emb.size(0), -1)  # [B,D]
        enhanced_graph_emb = torch.cat([graph_emb, ring_mean], dim=-1)  # [B, 2D]

        # === 对比 ===
        batch_size = enhanced_graph_emb.size(0)
        half = enhanced_graph_emb.size(0) // 2
        idx = torch.cat([
            torch.arange(half, enhanced_graph_emb.size(0)),
            torch.arange(0, half)
        ], dim=0)
        shuf_emb = enhanced_graph_emb[idx]

        lbl = torch.cat([
            torch.ones(enhanced_graph_emb.size(0)),
            torch.zeros(enhanced_graph_emb.size(0))
        ], dim=0).cuda()
        print("enhanced_graph_emb:", enhanced_graph_emb.shape)
        print("enhanced_graph_emb.requires_grad:", enhanced_graph_emb.requires_grad)
        print("shuf_emb:", shuf_emb.shape)
        print("shuf_emb.requires_grad:", shuf_emb.requires_grad)
        print("lbl:", lbl)
        with torch.no_grad():
            pos_sim = F.cosine_similarity(enhanced_graph_emb, enhanced_graph_emb).mean()
            neg_sim = F.cosine_similarity(enhanced_graph_emb, shuf_emb).mean()
            print(f"Pos pair mean cosine: {pos_sim:.4f}, Neg pair mean cosine: {neg_sim:.4f}")

        # main_loss = F.cosine_embedding_loss(
        #     torch.cat([enhanced_graph_emb, enhanced_graph_emb], dim=0),
        #     torch.cat([enhanced_graph_emb, shuf_emb], dim=0),
        #     lbl, margin=0.5
        # )

        main_loss = nt_xent(enhanced_graph_emb, shuf_emb, temperature=0.5)



        loss = main_loss + 0.8 * ring_loss
        # loss = main_loss + 0.8 * ring_loss

        total_loss += loss.item()
        total_main_loss += main_loss.item()
        total_ring_loss += ring_loss.item()
        loss.backward()
        optimiser.step()

        # if step == 0:
        #     print(f"[Debug] step {step} | main_loss: {main_loss.item():.4f} | ring_loss: {ring_loss.item():.4f}")

    avg_loss = total_loss / (step + 1)
    avg_main_loss = total_main_loss / (step + 1)
    avg_ring_loss = total_ring_loss / (step + 1)

    print(
        f" Epoch [{epoch}] | avg_loss: {avg_loss:.4f} | main_loss: {avg_main_loss:.4f} | ring_loss: {avg_ring_loss:.4f}")

    if avg_loss < best:
        best = avg_loss
        cnt_wait = 0
        torch.save(model.state_dict(), args.save_name)
        print(f" Model saved at epoch {epoch}")

        if args.generate_fewshot:
            print(f" Generating {args.shotnum}-shot support sets ({args.test_times} tasks)...")
            for i in range(args.test_times):
                np.random.seed(args.seed + i)
                torch.manual_seed(args.seed + i)
                torch.cuda.manual_seed(args.seed + i)

                indices = []
                labels_all = np.array(dataset.data.y)
                for cls in range(class_num):
                    cls_indices = np.where(labels_all == cls)[0]

                    if len(cls_indices) == 0:
                        print(f"[Few-shot] Class {cls} has no samples, skipping.")
                        continue

                    if len(cls_indices) < args.shotnum:
                        print(
                            f"[Few-shot] Class {cls} has only {len(cls_indices)} samples, sampling with replacement.")
                        chosen = np.random.choice(cls_indices, size=args.shotnum, replace=True)
                        indices.extend(chosen)
                        continue

                    # === 多样性采样 ===
                    degrees = []
                    for idx in cls_indices:
                        graph = dataset[int(idx)]
                        degrees.append(graph.edge_index.size(1) // 2)
                    degrees = np.array(degrees)
                    sorted_indices = cls_indices[np.argsort(degrees)]

                    chosen = [
                        sorted_indices[0],
                        sorted_indices[len(sorted_indices) // 4],
                        sorted_indices[len(sorted_indices) // 2],
                        sorted_indices[3 * len(sorted_indices) // 4],
                        sorted_indices[-1]
                    ]

                    indices.extend(chosen)
                    # 正确打印 degrees
                    chosen_idx_in_sorted = [
                        0, len(sorted_indices) // 4, len(sorted_indices) // 2, 3 * len(sorted_indices) // 4, -1
                    ]
                    print(f"[Few-shot] Class {cls}: chosen degrees = {degrees[chosen_idx_in_sorted].tolist()}")

                support_subset = [dataset[int(idx)] for idx in indices]
                support_features, support_adj, support_labels, support_complex, support_batch = process.process_tu(
                    support_subset, class_num, ft_size
                )

                #  如果是 scipy.sparse，记得转成 Tensor
                if hasattr(support_adj, "todense"):
                    support_adj = torch.FloatTensor(support_adj.todense())
                support_features = torch.FloatTensor(support_features)

                graph_len = [data.x.shape[0] for data in support_subset]

                os.makedirs(f"data/fewshot_{args.dataset}_graph/{args.shotnum}shot_{args.dataset}_graph/{i}/",
                            exist_ok=True)
                torch.save(support_adj,
                           f"data/fewshot_{args.dataset}_graph/{args.shotnum}shot_{args.dataset}_graph/{i}/adj.pt")
                torch.save(support_features,
                           f"data/fewshot_{args.dataset}_graph/{args.shotnum}shot_{args.dataset}_graph/{i}/feature.pt")
                torch.save(torch.LongTensor([data.y[0].item() for data in support_subset]),
                           f"data/fewshot_{args.dataset}_graph/{args.shotnum}shot_{args.dataset}_graph/{i}/labels.pt")
                torch.save(torch.LongTensor(graph_len),
                           f"data/fewshot_{args.dataset}_graph/{args.shotnum}shot_{args.dataset}_graph/{i}/graph_len.pt")
                torch.save(support_batch, f"data/fewshot_{args.dataset}_graph/{args.shotnum}shot_{args.dataset}_graph/{i}/batch.pt")

                if i == 1:
                    os.makedirs(f"data/fewshot_{args.dataset}_graph/testset/", exist_ok=True)
                    torch.save(support_adj, f"data/fewshot_{args.dataset}_graph/testset/adj.pt")
                    torch.save(support_features, f"data/fewshot_{args.dataset}_graph/testset/feature.pt")
                    torch.save(torch.LongTensor([data.y[0].item() for data in support_subset]),
                               f"data/fewshot_{args.dataset}_graph/testset/labels.pt")
                    torch.save(torch.LongTensor(graph_len),
                               f"data/fewshot_{args.dataset}_graph/testset/graph_len.pt")
                    torch.save(support_batch,
                       f"data/fewshot_{args.dataset}_graph/{args.shotnum}shot_{args.dataset}_graph/{i}/batch.pt")
            print(f" Saved support set {i} with {len(indices)} samples.")
    else:
        cnt_wait += 1
        print(f"cnt_wait: {cnt_wait}")

    if cnt_wait >= patience:
     print("Early stopping!")
     break
