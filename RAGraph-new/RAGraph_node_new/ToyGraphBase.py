import torch
import torch.nn.functional as F

from torch import Tensor
# from torch_geometric.loader import DataLoader
# from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

from .complex import Cochain,Complex,CochainBatch,ComplexBatch
from .Augmentation import Augmentation
from .InverseSampling import InverseSampling
from .Propagation import Propagation
from .PositionAwareEncoder import PositionAwareEncoder
from .SimilarityFunctions import SimilarityFunctions
from .utility import process_tu_dataset
from .helper_test import get_rings
import networkx as nx
import numpy as np


class ToyGraphBase:
    def __init__(self, pretrain_model, num_class, emb_size, query_graph_hop) -> None:
        # construct phase
        self.num_inverse_sample = 10  # set 0 to disable inverse sampling
        self.num_augment_scale = 3  # set 0 to disable augment

        # inference phase
        self.retrieve_num = num_class + 1
        self.noise_retrieve_num = 1

        self.num_anchors = 10  # log2(19580)
        self.dis_q = 10

        self.structure_weight = 0.0
        self.semantic_weight = 0.999

        # resources
        self.toy_graph_hop = query_graph_hop - 1  # RAG is also 1 hop
        self.pretrain_model = pretrain_model
        # 6.12备注：原图构造嵌入sample-based
        self.resource_keys = torch.empty(size=(0, emb_size)).cuda()
        self.resource_values = torch.empty(size=(0, emb_size)).cuda()
        self.resource_labels = torch.empty(size=(0, num_class)).cuda()
        self.resource_positions = torch.empty(size=(0, self.num_anchors)).cuda()
        # 6.12备注：Complex结构构造嵌入complex-based
        self.resource_keys_complex = torch.empty(size=(0, emb_size)).cuda()
        self.resource_values_complex = torch.empty(size=(0, emb_size)).cuda()
        self.resource_labels_complex = torch.empty(size=(0, num_class)).cuda()

    def build_toy_graph(self, resource_dataset: TUDataset):
        num_node_attributes = resource_dataset.num_node_attributes
        resource_loader = DataLoader(resource_dataset, batch_size=1, shuffle=False)
        for data in resource_loader:
            try:
                features, adj, node_labels = process_tu_dataset(data, num_node_attributes)
            except ValueError as exc:
                print(f"[ToyGraphBase] skip invalid resource batch: {exc}")
                continue
            self._build_toy_graph_base(features, adj, node_labels)

    def retrieve(self, search_keys: Tensor, search_adj: Tensor, add_noise: bool):
        # (query_num, resource_num)
        # search_positions = PositionAwareEncoder.encode_position_aware_code(search_adj, self.num_anchors, self.dis_q)
        # structure_similarities = SimilarityFunctions.calculate_cosine_similarity(search_positions, self.resource_positions)

        # (query_num, resource_num)
        semantic_similarities = SimilarityFunctions.calculate_cosine_similarity(search_keys, self.resource_keys)

        # (1, num_metric)
        # similarity_weights = torch.tensor([[self.structure_weight, self.semantic_weight]]).cuda()
        # (num_metric, query_num, resource_num)
        # similarity_matrices = torch.stack([structure_similarities, semantic_similarities], dim=0)

        # (query_num, num_metric)
        # similarity_scores = torch.einsum('ij,jkl->ikl', similarity_weights, similarity_matrices).squeeze(0)

        similarity_scores = semantic_similarities

        # Get the top-k scores and their indices for each query in one operation
        retrieve_num = 2 * self.retrieve_num if add_noise else self.retrieve_num
        topk_scores, topk_indices = torch.topk(similarity_scores, retrieve_num, largest=True, sorted=True)

        # Retrieve the embeddings and labels corresponding to the top-k indices
        rag_embeddings = self.resource_values[topk_indices]  # (query_num, topk, emb_size)
        rag_labels = self.resource_labels[topk_indices]  # (query_num, topk, 1)

        if add_noise:
            noise_indices = torch.randint(0, self.resource_values.shape[0],
                                          (search_keys.shape[0], self.noise_retrieve_num))

            noise_rag_embeddings = self.resource_values[noise_indices]
            noise_rag_labels = self.resource_labels[noise_indices]
            rag_embeddings = torch.cat([rag_embeddings, noise_rag_embeddings], dim=1)
            rag_labels = torch.cat([rag_labels, noise_rag_labels], dim=1)

        return rag_embeddings, rag_labels

    def show(self):
        print('resource_keys', self.resource_keys.shape)
        print('resource_values', self.resource_values.shape)
        print('resource_labels', self.resource_labels.shape)
        print('resource positions', self.resource_positions.shape)

        print("label count distribution", torch.sum(self.resource_labels, dim=0))

    def _build_toy_graph_base(self, features, adj, node_labels):
        device = next(self.pretrain_model.parameters()).device
        for aug_features, aug_adj in Augmentation.augment_graph(self.num_augment_scale, features, adj):
            # 强制去掉 batch 维度，确保输入为 [N, F] 和 [N, N]
            if aug_features.dim() == 3:
                aug_features = aug_features.squeeze(0)
            if aug_adj.dim() == 3:
                aug_adj = aug_adj.squeeze(0)

            # 确保特征和邻接矩阵在正确设备和类型
            aug_features = aug_features.to(device).float()
            aug_adj = aug_adj.to(device).float()
            node_labels = node_labels.to(device)
            # 原始 GNN 推理
            embeddings = self.pretrain_model.inference(aug_features, aug_adj)

            # 原始 Sample-Based 表征
            if self.num_inverse_sample > 0:
                sample_prob = InverseSampling.compute_sample_prob(aug_adj)
                sample_mask = torch.multinomial(sample_prob, num_samples=self.num_inverse_sample, replacement=True)
                sample_adj = aug_adj[sample_mask, :][:, sample_mask]
                sample_keys: Tensor = embeddings[sample_mask]
                sample_labels = node_labels[sample_mask]
            else:
                sample_adj = aug_adj
                sample_keys: Tensor = embeddings
                sample_labels = node_labels

            sample_keys = F.normalize(sample_keys, p=2, dim=-1)
            sample_values = Propagation.aggregate_k_hop_features(sample_adj, sample_keys, self.toy_graph_hop)
            sample_positions = PositionAwareEncoder.encode_position_aware_code(sample_adj, self.num_anchors, self.dis_q)

            # Complex 构造部分
            try:
                # 确保邻接矩阵是稀疏格式
                edge_index = aug_adj.to_sparse().indices().long()

                # 构建 0维 Cochain（节点）
                v_cochain = Cochain(
                    dim=0,
                    x=aug_features,
                    y=node_labels
                )

                # 构建 1维 Cochain（边）
                edge_attr = torch.ones(edge_index.size(1), 1, device=aug_adj.device)
                e_cochain = Cochain(
                    dim=1,
                    x=edge_attr,
                    boundary_index=edge_index  # 使用稀疏边索引
                )

                # 创建 Complex
                complex_obj = Complex(v_cochain, e_cochain)

                # Complex 级别嵌入
                complex_embed = self.pretrain_model.inference(complex_obj)
                if complex_embed.dim() == 3:
                    complex_embed = complex_embed.squeeze(0)
                complex_keys = F.normalize(complex_embed, p=2, dim=-1)
                complex_values = Propagation.aggregate_k_hop_features(
                    aug_adj,
                    complex_keys,
                    self.toy_graph_hop
                )
                complex_labels = node_labels

            except Exception as e:
                print(f"[Complex 构造失败] {str(e)}")
                print(f"特征形状: {aug_features.shape}, 邻接矩阵形状: {aug_adj.shape}")
                complex_keys = torch.empty((0, sample_keys.size(1)), device=sample_keys.device)
                complex_values = torch.empty_like(complex_keys)
                complex_labels = torch.empty((0, node_labels.size(1)), device=node_labels.device)

            # 合并表征
            self.resource_keys = torch.cat([self.resource_keys, sample_keys, complex_keys], dim=0)
            self.resource_values = torch.cat([self.resource_values, sample_values, complex_values], dim=0)
            self.resource_labels = torch.cat([self.resource_labels, sample_labels, complex_labels], dim=0)
            self.resource_positions = torch.cat([self.resource_positions, sample_positions], dim=0)
