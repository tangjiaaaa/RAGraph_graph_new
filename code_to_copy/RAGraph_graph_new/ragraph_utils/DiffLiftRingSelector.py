import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def rings_from_boundary(ring_boundary, edge_boundary, num_nodes):
    """Recover candidate 2-cell node sets from existing RAGraph Complex data."""
    if ring_boundary is None or edge_boundary is None or ring_boundary.numel() == 0:
        return []

    rings = {}
    for col in range(ring_boundary.size(1)):
        edge_id = int(ring_boundary[0, col].item())
        ring_id = int(ring_boundary[1, col].item())
        if edge_id < 0:
            continue

        nodes = None
        incident = edge_boundary[1] == edge_id
        if incident.any():
            node_ids = edge_boundary[0, incident].unique()
            if node_ids.numel() >= 2:
                nodes = node_ids[:2]

        if nodes is None and edge_id < edge_boundary.size(1):
            nodes = edge_boundary[:, edge_id]

        if nodes is None:
            continue

        u, v = int(nodes[0].item()), int(nodes[1].item())
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            rings.setdefault(ring_id, set()).update([u, v])

    return [sorted(nodes) for _, nodes in sorted(rings.items()) if len(nodes) >= 2]


class DiffLiftRingSelector(nn.Module):
    """DiffLift-style differentiable selection over candidate rings.

    The original ReTAG/RAGraph code already constructs candidate 2-cells. This
    module does not rebuild the complex; it learns which candidate rings should
    contribute to the cellular retrieval key.
    """

    def __init__(self, emb_size, hidden_size=None, temperature=1.0, hard=False, sparsity_weight=1e-3):
        super().__init__()
        hidden_size = hidden_size or emb_size
        self.temperature = temperature
        self.hard = hard
        self.sparsity_weight = sparsity_weight
        self.cell_mlp = nn.Sequential(
            nn.Linear(emb_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, node_emb, ring_boundary, edge_boundary):
        rings = rings_from_boundary(ring_boundary, edge_boundary, node_emb.size(0))
        return self.forward_rings(node_emb, rings)

    def forward_rings(self, node_emb, rings):
        device = node_emb.device
        emb_size = node_emb.size(-1)
        if len(rings) == 0:
            zero = torch.zeros(emb_size, device=device)
            return zero, zero.sum(), {
                "ring_prob": torch.empty(0, device=device),
                "ring_count": 0,
                "selected_mass": zero.sum(),
            }

        ring_tensors = [torch.tensor(ring, dtype=torch.long, device=device) for ring in rings]
        ring_idx = pad_sequence(ring_tensors, batch_first=True, padding_value=-1)
        mask = ring_idx.ne(-1).unsqueeze(-1)
        gathered = node_emb[ring_idx.clamp_min(0)]
        ring_emb = (gathered * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

        logits = self.cell_mlp(ring_emb)
        if self.training:
            sample = F.gumbel_softmax(logits, tau=self.temperature, hard=self.hard, dim=-1)
            weights = sample[:, 1]
        else:
            weights = torch.softmax(logits, dim=-1)[:, 1]

        ring_mean = (weights.unsqueeze(-1) * ring_emb).sum(dim=0) / weights.sum().clamp_min(1e-6)
        aux_loss = self.sparsity_weight * weights.mean()
        return ring_mean, aux_loss, {
            "ring_prob": torch.softmax(logits, dim=-1)[:, 1],
            "ring_count": len(rings),
            "selected_mass": weights.sum().detach(),
        }
