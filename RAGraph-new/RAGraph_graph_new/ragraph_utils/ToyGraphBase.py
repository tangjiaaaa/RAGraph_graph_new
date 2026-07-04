import torch
import torch.nn.functional as F
import random

from torch import Tensor
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset

from .complex import Cochain, Complex, CochainBatch, ComplexBatch
from .Augmentation import Augmentation
from .InverseSampling import InverseSampling
from .Propagation import Propagation
from .PositionAwareEncoder import PositionAwareEncoder
from .SimilarityFunctions import SimilarityFunctions
from .utility import process_tu_dataset
from .TaskDecoder import TaskDecoder
from .extract_ring import extract_ring_mean, ring_contrastive_loss


class ToyGraphBase:
    def __init__(self, pretrain_model, num_class, emb_size, query_graph_hop, max_ring=10):
        self.pretrain_model = pretrain_model
        self.num_class = num_class
        self.emb_size = emb_size
        self.query_graph_hop = query_graph_hop
        self.toy_graph_hop = query_graph_hop
        self.max_ring = max_ring
        self.retrieve_num = 5  # 默认检索 Top-5
        self.num_augment_scale = 1
        self.structure_weight = 0.5
        self.semantic_weight = 0.4
        self.ring_weight = 0.1
        self.utility_weight = 0.0
        self.utility_momentum = 0.9
        self.num_anchors = 16
        self.dis_q = 0.1
        # 以下变量需要在构图后赋值
        self.resource_keys = None
        self.resource_values = None
        self.resource_labels = None
        self.resource_positions = None
        self.resource_ring_feats = None
        self.resource_utility = None

    def build_toy_graph(self, resource_dataset: TUDataset):
        self.resource_keys = []
        self.resource_values = []
        self.resource_labels = []
        self.resource_positions = []
        self.resource_complexes = []
        self.resource_ring_feats = []

        num_node_attributes = resource_dataset.num_node_attributes
        resource_loader = DataLoader(resource_dataset, batch_size=1, shuffle=False)

        for data in resource_loader:
            features, adj, graph_labels, complex_batch, _ = process_tu_dataset(data, self.num_class,
                                                                               num_node_attributes,
                                                                               max_ring=self.max_ring)
            features = features.cuda()
            adj = adj.cuda()
            label = F.one_hot(graph_labels, num_classes=self.num_class).float()
            for aug_features, aug_adj in Augmentation.augment_graph(self.num_augment_scale, features, adj):
                self._build_toy_graph_base(aug_features, aug_adj, label, complex_batch)

        self.resource_keys = torch.cat(self.resource_keys, dim=0)
        self.resource_values = torch.cat(self.resource_values, dim=0)
        self.resource_labels = torch.cat(self.resource_labels, dim=0)
        self.resource_positions = torch.cat(self.resource_positions, dim=0)
        self.resource_ring_feats = torch.cat(self.resource_ring_feats, dim=0)
        self.resource_utility = torch.zeros(self.resource_keys.size(0), device=self.resource_keys.device)

    def _build_toy_graph_base(self, features, adj, label, complex_batch):
        device = next(self.pretrain_model.parameters()).device

        features = features.to(device).float()
        adj = adj.to(device).float()
        label = label.to(device)
        node_emb, _ = self.pretrain_model.embed(features, adj)
        # === Graph-level embedding ===
        graph_embed = node_emb.mean(dim=0, keepdim=True)

        if graph_embed.dim() == 3:
            graph_embed = graph_embed.mean(dim=1)
        if graph_embed.dim() == 2 and graph_embed.size(0) > 1:
            graph_embed = graph_embed.mean(dim=0, keepdim=True)
        elif graph_embed.dim() == 1:
            graph_embed = graph_embed.unsqueeze(0)

        ring_mean = extract_ring_mean(node_emb, complex_batch)
        ring_mean = ring_mean.unsqueeze(0).expand(graph_embed.size(0), -1)
        enhanced_emb = torch.cat([graph_embed, ring_mean], dim=-1)
        key = F.normalize(enhanced_emb, p=2, dim=-1)

        #  K-hop 聚合
        use_khop = True
        if use_khop and self.toy_graph_hop > 0:
            value = Propagation.aggregate_k_hop_features(adj, features, self.toy_graph_hop)  # [N,D]
            value = value.mean(dim=0, keepdim=True)  # [1,D]
        else:
            value = graph_embed

        #  保证 ring_mean 至少是 [1, D]
        if ring_mean.dim() == 1:
            ring_mean = ring_mean.unsqueeze(0)

        #  保证 ring_mean 的最后一维是 D
        if ring_mean.size(1) != node_emb.size(1):
            # print(f"[DEBUG] Fix ring_mean shape: {ring_mean.shape} → [:,{node_emb.size(1)}]")
            ring_mean = ring_mean[:, :node_emb.size(1)]

        assert ring_mean.size(1) == node_emb.size(1), f"[ToyGraphBase] ring_mean final wrong shape: {ring_mean.shape}"

        # 同理 value 聚合后必须是 [1, D]
        if value.size(1) != node_emb.size(1):
            value = graph_embed.clone()

        assert value.size(1) == node_emb.size(1), f"[ToyGraphBase] value final wrong shape: {value.shape}"

        # 拼接: 保证 [1, 2D]
        value = torch.cat([value, ring_mean], dim=-1)

        assert value.size(1) == 2 * node_emb.size(1), f"[ToyGraphBase] value shape after concat wrong: {value.shape}"

        pos = PositionAwareEncoder.encode_position_aware_code(adj, self.num_anchors, self.dis_q)
        pos = pos.mean(dim=0, keepdim=True)  # [1,num_anchors]

        key = key.detach().clone()
        value = value.detach().clone()

        self.resource_keys.append(key)
        self.resource_values.append(value)
        self.resource_labels.append(label)
        self.resource_positions.append(pos)
        self.resource_ring_feats.append(ring_mean.detach().clone())

    def retrieve(self, search_keys: Tensor, search_adj: Tensor, complex_batch, search_ring_feat: Tensor,
                 add_noise: bool, return_indices: bool = False):
        B = search_keys.size(0)

        # === semantic similarity ===
        semantic_sim = SimilarityFunctions.calculate_cosine_similarity(
            search_keys, self.resource_keys
        )  # [B, num_toy]

        # === structure similarity ===
        if search_adj.dim() == 2:
            pos = PositionAwareEncoder.encode_position_aware_code(
                search_adj, self.num_anchors, self.dis_q
            )
            pos = pos.mean(dim=0, keepdim=True)
            search_positions = pos.repeat(B, 1) if B > 1 else pos
        elif search_adj.dim() == 3:
            search_positions = PositionAwareEncoder.encode_batch_graph_signature(
                search_adj, self.num_anchors, self.dis_q
            )
        else:
            raise ValueError(f"[ERROR] search_adj must be [N,N] or [B,N,N], got {search_adj.shape}")

        structure_sim = SimilarityFunctions.calculate_cosine_similarity(
            search_positions, self.resource_positions
        )  # [B, num_toy]

        # === ring similarity ===
        ring_sim = SimilarityFunctions.calculate_cosine_similarity(
            search_ring_feat, self.resource_ring_feats
        )  # [B, num_toy]

        # === combine similarity ===
        similarity_weights = torch.tensor(
            [self.structure_weight, self.semantic_weight, self.ring_weight],
            device=semantic_sim.device
        )
        similarity_matrices = torch.stack([structure_sim, semantic_sim, ring_sim], dim=0)
        similarity_scores = torch.einsum('i,ijk->jk', similarity_weights, similarity_matrices)  # [B, num_toy]
        if self.resource_utility is not None and self.utility_weight > 0:
            utility_bias = self.resource_utility.unsqueeze(0).expand_as(similarity_scores)
            similarity_scores = similarity_scores + self.utility_weight * utility_bias

        # === 修复重点：动态调整 retrieve_num ===
        retrieve_num = 2 * self.retrieve_num if add_noise else self.retrieve_num

        # 核心修复：确保要检索的数量不大于数据库里的总样本数
        max_available = similarity_scores.size(1)
        if retrieve_num > max_available:
            # print(f"[WARN] 样本不足: 请求 {retrieve_num}, 实际只有 {max_available}. 自动调整为 {max_available}.")
            retrieve_num = max_available

        topk_scores, topk_indices = torch.topk(similarity_scores, retrieve_num, largest=True, sorted=True)
        rag_weights = torch.softmax(topk_scores, dim=-1)  # [B, K]

        rag_embeddings = self.resource_values[topk_indices]  # [B, K, D]
        rag_labels = self.resource_labels[topk_indices]  # [B, K, num_class]

        if add_noise:
            rag_embeddings = self._add_noise(rag_embeddings)

        if return_indices:
            return rag_embeddings, rag_labels, rag_weights, topk_indices
        return rag_embeddings, rag_labels, rag_weights

    @torch.no_grad()
    def update_memory_utility(self, topk_indices: Tensor, target_labels: Tensor, rag_weights: Tensor):
        """Update memory usefulness from training-label feedback.

        This is a supervised memory calibration signal: memories whose labels
        match the current training graph are upweighted; mismatched memories are
        downweighted in proportion to their retrieval weights.
        """
        if self.resource_utility is None:
            return
        target = target_labels.view(-1, 1).to(topk_indices.device)
        retrieved_y = self.resource_labels[topk_indices].argmax(dim=-1)
        match = retrieved_y.eq(target).float()
        delta = (2.0 * match - 1.0) * rag_weights.detach()
        flat_idx = topk_indices.reshape(-1)
        flat_delta = delta.reshape(-1).to(self.resource_utility.device)

        old = self.resource_utility[flat_idx]
        new = self.utility_momentum * old + (1.0 - self.utility_momentum) * flat_delta
        self.resource_utility[flat_idx] = new.clamp(min=-1.0, max=1.0)

    def show(self):
        print("[ToyGraphBase Summary]")
        print("resource_keys:", self.resource_keys.shape)
        print("resource_positions:", self.resource_positions.shape)
        print("resource_labels:", self.resource_labels.shape)

        N = self.resource_keys.size(0)
        # 防止 show 的时候样本太少报错
        sample_size = min(500, N)
        if sample_size < 2:
            print("样本太少，无法计算 Cosine 统计。")
            return

        sample_idx = torch.randint(0, N, (sample_size, 2), device=self.resource_keys.device)
        vec1 = self.resource_keys[sample_idx[:, 0]]
        vec2 = self.resource_keys[sample_idx[:, 1]]
        cos_sim = F.cosine_similarity(vec1, vec2, dim=-1)

        print("Sampled Proto Cosine mean:", cos_sim.mean().item(),
              "min:", cos_sim.min().item(),
              "max:", cos_sim.max().item())

    def _add_noise(self, embeddings):
        noise_std = self.noise_std if hasattr(self, 'noise_std') else 0.05
        noise = torch.randn_like(embeddings) * noise_std
        noisy_embeddings = embeddings + noise
        return noisy_embeddings
