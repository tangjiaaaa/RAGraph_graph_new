import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskAwareReranker(nn.Module):
    """Query-conditioned reranker for retrieved cellular memories."""

    def __init__(self, query_dim, candidate_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(query_dim + candidate_dim + 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, query, candidates, base_weights):
        bsz, topk, _ = candidates.shape
        query_expand = query.unsqueeze(1).expand(-1, topk, -1)
        query_for_candidate = query_expand[..., : candidates.size(-1)]

        cosine = F.cosine_similarity(
            F.normalize(query_for_candidate, dim=-1),
            F.normalize(candidates, dim=-1),
            dim=-1,
        ).unsqueeze(-1)
        l2 = (query_for_candidate - candidates).pow(2).mean(dim=-1, keepdim=True)
        base = base_weights.unsqueeze(-1)
        rank_pos = torch.linspace(0, 1, topk, device=query.device).view(1, topk, 1).expand(bsz, -1, -1)

        score_input = torch.cat([query_expand, candidates, cosine, l2, base, rank_pos], dim=-1)
        utility_logits = self.scorer(score_input).squeeze(-1)
        rerank_weights = torch.softmax(utility_logits + torch.log(base_weights.clamp_min(1e-8)), dim=-1)
        return rerank_weights, utility_logits


def retrieval_alignment_loss(utility_logits, candidate_labels, target_labels):
    """Train the reranker to upweight memories whose labels match the query."""
    candidate_y = candidate_labels.argmax(dim=-1)
    target = target_labels.view(-1, 1).expand_as(candidate_y)
    helpful = candidate_y.eq(target).float()
    return F.binary_cross_entropy_with_logits(utility_logits, helpful)
