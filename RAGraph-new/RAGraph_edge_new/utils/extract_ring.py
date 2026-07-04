import torch
import random
import torch.nn.functional as F
def ring_contrastive_loss_new(node_emb, ring_boundary, edge_index, temperature=0.5):
    if ring_boundary.size(1) == 0:
        return torch.tensor(0.0, device=node_emb.device), \
               torch.zeros(node_emb.size(1), device=node_emb.device)

    ring_dict = {}
    for i in range(ring_boundary.size(1)):
        edge_id = ring_boundary[0, i].item()
        if edge_id >= edge_index.size(1):
            continue
        u = edge_index[0, edge_id].item()
        v = edge_index[1, edge_id].item()
        if u >= node_emb.size(0) or v >= node_emb.size(0):
            continue
        ring_id = ring_boundary[1, i].item()
        ring_dict.setdefault(ring_id, set()).update([u, v])

    ring_views_1, ring_views_2 = [], []
    for node_set in ring_dict.values():
        if len(node_set) < 2:
            continue
        nodes = list(node_set)
        # view1
        sampled1 = torch.tensor(random.sample(nodes, max(2, len(nodes)//2))).to(node_emb.device)
        # view2
        sampled2 = torch.tensor(random.sample(nodes, max(2, len(nodes)//2))).to(node_emb.device)

        h1 = node_emb[sampled1].mean(dim=0)
        h2 = node_emb[sampled2].mean(dim=0)

        ring_views_1.append(h1)
        ring_views_2.append(h2)

    if len(ring_views_1) < 2:
        return torch.tensor(0.0, device=node_emb.device), \
               torch.zeros(node_emb.size(1), device=node_emb.device)

    z1 = torch.stack(ring_views_1)  # [N, D]
    z2 = torch.stack(ring_views_2)  # [N, D]
    N, D = z1.size()

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    representations = torch.cat([z1, z2], dim=0)  # [2N, D]
    similarity_matrix = torch.matmul(representations, representations.T)  # [2N, 2N]

    labels = torch.cat([torch.arange(N) + N, torch.arange(N)], dim=0).to(node_emb.device)  # [2N]
    mask = torch.eye(2*N, dtype=torch.bool, device=node_emb.device)

    logits = similarity_matrix / temperature
    logits = logits.masked_fill(mask, -9e15)  # remove self-similarity

    loss = F.cross_entropy(logits, labels)

    ring_mean = (z1 + z2).mean(dim=0)
    return loss, ring_mean
def ring_contrastive_loss(node_emb, ring_boundary, edge_index):
    if ring_boundary.size(1) == 0:
        return torch.tensor(0.0, device=node_emb.device, requires_grad=True), \
               torch.zeros(node_emb.size(1), device=node_emb.device, requires_grad=True)

    ring_dict = {}
    for i in range(ring_boundary.size(1)):
        edge_id = ring_boundary[0, i].item()
        if edge_id >= edge_index.size(1):
            continue
        u = edge_index[0, edge_id].item()
        v = edge_index[1, edge_id].item()
        if u >= node_emb.size(0) or v >= node_emb.size(0):
            continue
        ring_id = ring_boundary[1, i].item()
        ring_dict.setdefault(ring_id, set()).update([u, v])

    ring_embeddings = []
    for node_set in ring_dict.values():
        if len(node_set) < 2:
            continue
        nodes = list(node_set)

        if len(nodes) > 3:
            sampled_nodes = torch.tensor(random.sample(nodes, len(nodes) - 1)).to(node_emb.device)
        else:
            sampled_nodes = torch.tensor(nodes).to(node_emb.device)

        ring_emb = node_emb[sampled_nodes].mean(dim=0)
        ring_emb = ring_emb + 0.05 * torch.randn_like(ring_emb)  # 小扰动，防 collapse
        ring_embeddings.append(ring_emb)

    if len(ring_embeddings) < 2:
        return torch.tensor(0.0, device=node_emb.device, requires_grad=True), \
               torch.zeros(node_emb.size(1), device=node_emb.device, requires_grad=True)

    ring_embeddings = torch.stack(ring_embeddings, dim=0)  # [R, D]
    anchor = ring_embeddings
    positive = torch.roll(ring_embeddings, shifts=1, dims=0)
    target = torch.ones(anchor.size(0), device=node_emb.device)

    loss = F.cosine_embedding_loss(anchor, positive, target, margin=0.3)
    ring_mean = ring_embeddings.mean(dim=0)
    if ring_mean.size(0) != node_emb.size(1):
        ring_mean = ring_mean[:node_emb.size(1)]
    # print("ring_mean.grad:", ring_mean.grad)
    return loss, ring_mean


def extract_ring_mean(node_emb, complex_obj):
    cochain_2 = complex_obj.cochains[2] if 2 in complex_obj.cochains else None
    cochain_1 = complex_obj.cochains[1] if 1 in complex_obj.cochains else None

    if (cochain_2 is None or
        cochain_2.boundary_index is None or
        cochain_2.boundary_index.size(1) == 0 or
        cochain_1 is None or
        cochain_1.boundary_index is None or
        cochain_1.boundary_index.size(1) == 0):
        return torch.zeros(node_emb.size(1), device=node_emb.device, requires_grad=True)

    _, ring_mean = ring_contrastive_loss(
        node_emb,
        cochain_2.boundary_index.cuda(),
        cochain_1.boundary_index.cuda()
    )
    return ring_mean




def extract_ring_mean_batch(node_emb, complex_batch, batch_vector):
    """
    针对 ComplexBatch + node_emb，提取每张图的 ring_mean 并聚合
    """
    ring_means = []
    num_graphs = batch_vector.max().item() + 1

    for i in range(num_graphs):
        node_emb_i = node_emb[batch_vector == i]
        complex_obj = complex_batch[i]

        if complex_obj is None or \
           2 not in complex_obj.cochains or \
           complex_obj.cochains[2] is None or \
           complex_obj.cochains[2].boundary_index is None or \
           complex_obj.cochains[2].boundary_index.size(1) == 0:
            ring_mean_i = torch.zeros(node_emb.size(1), device=node_emb.device, requires_grad=True)
            continue
        else:
            _, ring_mean_i = ring_contrastive_loss(
                node_emb_i,
                complex_obj.cochains[2].boundary_index.cuda(),
                complex_obj.cochains[1].boundary_index.cuda()
            )

        ring_means.append(ring_mean_i)

    if ring_means:
        return torch.stack(ring_means, dim=0)
    else:
        return torch.zeros((1, node_emb.size(1)), device=node_emb.device, requires_grad=True)


# def extract_ring_mean(node_emb, complex_obj):
#     if complex_obj.cochains[2] is not None:
#         ring_boundary = complex_obj.cochains[2].boundary_index.cuda()
#         edge_index = complex_obj.cochains[1].boundary_index.cuda()
#         _, ring_mean = ring_contrastive_loss(node_emb, ring_boundary, edge_index)
#     else:
#         ring_mean = torch.zeros(node_emb.size(1), device=node_emb.device, requires_grad=True)
#     ring_mean = ring_mean
#     # print("[DEBUG] ring_mean.requires_grad:", ring_mean.requires_grad)
#     # print("[DEBUG] ring_mean norm:", ring_mean.norm().item())
#
#     return ring_mean

# RAGraph的
# def ring_contrastive_loss(node_emb, ring_boundary, edge_index):
#     if ring_boundary.size(1) == 0:
#         print(" ring_boundary empty")
#         return torch.tensor(0.0, requires_grad=True).cuda(), torch.zeros(node_emb.size(1), requires_grad=True).cuda()
#
#     skipped_edgeid, skipped_nodeid = 0, 0
#     ring_dict = {}
#
#     for i in range(ring_boundary.size(1)):
#         edge_id = ring_boundary[0, i].item()
#         if edge_id >= edge_index.size(1):
#             skipped_edgeid += 1
#             continue
#         u = edge_index[0, edge_id].item()
#         v = edge_index[1, edge_id].item()
#         if u >= node_emb.size(0) or v >= node_emb.size(0):
#             skipped_nodeid += 1
#             continue
#         ring_id = ring_boundary[1, i].item()
#         ring_dict.setdefault(ring_id, set()).update([u, v])
#
#     ring_embeddings = []
#     for node_set in ring_dict.values():
#         if len(node_set) < 2:
#             continue
#         ring_emb = node_emb[list(node_set)].mean(dim=0)
#         ring_embeddings.append(ring_emb)
#
#     if len(ring_embeddings) < 2:
#         print(f"有效环数量不足，最终 ring_dict 有效: {len(ring_dict)}")
#         return torch.tensor(0.0, requires_grad=True).cuda(), torch.zeros(node_emb.size(1), requires_grad=True).cuda()
#
#     ring_embeddings = torch.stack(ring_embeddings)
#     # anchor = ring_embeddings
#     # positive = torch.roll(ring_embeddings, shifts=1, dims=0)
#     perm = torch.randperm(ring_embeddings.size(0))
#     anchor = ring_embeddings
#     positive = ring_embeddings[perm]
#     anchor = F.normalize(anchor, dim=-1)
#     positive = F.normalize(positive, dim=-1)
#     anchor = F.dropout(anchor, p=0.1, training=True)
#     positive = F.dropout(positive, p=0.1, training=True)
#
#     target = torch.ones(anchor.size(0)).to(anchor.device)
#     loss = F.cosine_embedding_loss(anchor, positive, target, margin=0.3)
#     # with torch.no_grad():
#     #     cos_sim = F.cosine_similarity(anchor, positive)
#     #     print(f"[ring_loss] mean={cos_sim.mean():.4f}, min={cos_sim.min():.4f}, max={cos_sim.max():.4f}")
#     ring_mean = ring_embeddings.mean(dim=0)
#     # print(f"[ring_loss] Valid rings: {len(ring_embeddings)}")
#
#     return loss, ring_mean
