import torch
import torch.nn.functional as F


class Propagation:
    @staticmethod
    def aggregate_k_hop_features(adj: torch.Tensor, x: torch.Tensor, k: int):
        aggregated_features = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        adj = torch.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)

        degree = adj.sum(dim=1, keepdim=True).clamp_min(1e-12)
        adj_normalized = torch.nan_to_num(adj / degree, nan=0.0, posinf=0.0, neginf=0.0)

        for _ in range(k):
            aggregated_features = torch.matmul(adj_normalized, aggregated_features)
            aggregated_features = torch.nan_to_num(aggregated_features, nan=0.0, posinf=0.0, neginf=0.0)
            aggregated_features = F.relu(aggregated_features)

        return aggregated_features
