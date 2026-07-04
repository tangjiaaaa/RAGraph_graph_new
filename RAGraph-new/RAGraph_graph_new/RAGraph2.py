import torch
import torch.nn as nn
import torch.nn.functional as F
from ragraph_utils import (
    ToyGraphBase,
    Propagation,
    FewShotBase,
    TaskDecoder,
    ring_contrastive_loss,
    ring_views_from_boundary,
    ring_contrastive_loss_from_views,
)
from torch_geometric.nn import global_mean_pool


class RAGraph(nn.Module):
    def __init__(self, pretrain_model, resource_dataset, feture_size, num_class, emb_size,
                 finetune=True, noise_finetune=False, dataset_name=None, ring_weight=0.1,
                 query_graph_hop=2, retrieve_num=5, fusion_gamma=0.2, max_ring=10) -> None:
        super(RAGraph, self).__init__()
        self.emb_size = emb_size
        self.num_class = num_class
        self.pretrain_model = pretrain_model
        self.ring_weight = ring_weight
        self.fusion_gamma = fusion_gamma
        self.ring_proj = nn.Sequential(
            nn.Linear(emb_size, emb_size * 2),
            nn.ReLU(),
            nn.Linear(emb_size * 2, emb_size)
        )
        self.dataset_name = dataset_name
        self.label_weight = fusion_gamma
        self.finetune = finetune
        self.noise_finetune = noise_finetune
        if self.noise_finetune:
            assert self.finetune

        self.query_graph_hop = query_graph_hop

        self.toy_graph_base = ToyGraphBase(pretrain_model, num_class, emb_size, self.query_graph_hop, max_ring=max_ring)
        self.toy_graph_base.retrieve_num = retrieve_num
        self.toy_graph_base.build_toy_graph(resource_dataset)
        self.toy_graph_base.noise_std = 0.05
        # decoder input: query_emb_pool + rag_embedding + ring_feat
        decoder_input_dim = emb_size + 2 * emb_size + emb_size
        self.decoder = TaskDecoder(decoder_input_dim, 512, num_class)
        self.graph_decoder = TaskDecoder(emb_size, 512, num_class)
        self.graph_logit_weight = 0.5
        self.reset_parameters()

        self.toy_graph_base.show()
        self.fewshot_base = FewShotBase(self.dataset_name, num_class, pretrain_model)

    def reset_parameters(self):
        self.decoder.reset_parameters()
        self.graph_decoder.reset_parameters()

    @staticmethod
    def _edge_index_from_complex(complex_obj):
        edge_cochain = complex_obj.cochains[1]
        return getattr(edge_cochain, "edge_index", edge_cochain.boundary_index)

    def forward(self, features, adj, complex_batch=None, batch=None, ptr=None, return_ring_loss=False):
        """
        前向传播（用于测试/推理阶段）。
        """
        node_emb, _ = self.pretrain_model.embed(features, adj)

        if batch is not None:
            graph_emb = global_mean_pool(node_emb, batch)
        elif ptr is not None:
            graph_emb = [node_emb[ptr[i]:ptr[i + 1]].mean(dim=0) for i in range(len(ptr) - 1)]
            graph_emb = torch.stack(graph_emb, dim=0)
        else:
            graph_emb = node_emb.mean(dim=0, keepdim=True)

        ring_loss = torch.tensor(0.0, device=node_emb.device)
        ring_means = []
        view1_all, view2_all = [], []

        # --- 计算 Ring 特征 ---
        if batch is not None:
            num_graphs = batch.max().item() + 1
            for i in range(num_graphs):
                node_emb_i = node_emb[batch == i]
                complex_obj = complex_batch[i] if complex_batch is not None else None

                if complex_obj is None or \
                        2 not in complex_obj.cochains or \
                        complex_obj.cochains[2] is None or \
                        complex_obj.cochains[2].boundary_index is None or \
                        complex_obj.cochains[2].boundary_index.size(1) == 0:
                    # 空 Ring 处理
                    ring_mean_i = torch.zeros(self.emb_size, device=node_emb.device)
                    if self.training:
                        ring_mean_i.requires_grad_(True)
                    single_loss = torch.tensor(0.0, device=node_emb.device)
                else:
                    view1_list, view2_list, ring_mean_i = ring_views_from_boundary(
                        node_emb_i,
                        complex_obj.cochains[2].boundary_index.to(node_emb.device),
                        self._edge_index_from_complex(complex_obj).to(node_emb.device),
                        noise_std=0.05 if self.training else 0.0
                    )
                    view1_all.extend(view1_list)
                    view2_all.extend(view2_list)
                    single_loss = torch.tensor(0.0, device=node_emb.device)

                ring_loss += single_loss
                ring_means.append(ring_mean_i)

            ring_loss = ring_contrastive_loss_from_views(
                view1_all, view2_all, device=node_emb.device
            )
            ring_mean = torch.stack(ring_means, dim=0)
        else:
            # 单图情况
            if complex_batch is not None and \
                    2 in complex_batch.cochains and \
                    complex_batch.cochains[2].boundary_index.size(1) > 0:
                single_loss, ring_mean = ring_contrastive_loss(
                    node_emb,
                    complex_batch.cochains[2].boundary_index.to(node_emb.device),
                    self._edge_index_from_complex(complex_batch).to(node_emb.device)
                )
                ring_loss = single_loss
                ring_mean = ring_mean.unsqueeze(0)
            else:
                ring_mean = torch.zeros(1, self.emb_size, device=node_emb.device)
                ring_loss = torch.tensor(0.0, device=node_emb.device)

        if ring_mean.requires_grad:
            ring_mean.retain_grad()

        if ring_mean.dim() == 1:
            ring_mean = ring_mean.unsqueeze(0)

        # --- RAG 检索 ---
        query_keys = torch.cat([graph_emb, ring_mean], dim=-1)
        add_noise = self.training and self.noise_finetune
        rag_embeddings, rag_labels, rag_weights = self.toy_graph_base.retrieve(
            query_keys, adj, complex_batch, ring_mean, add_noise
        )

        if self.finetune:
            attn_weights = rag_weights.unsqueeze(-1)
            rag_embedding = torch.sum(attn_weights * rag_embeddings, dim=1)
            rag_label = torch.sum(attn_weights * rag_labels, dim=1)

            query_embeddings = Propagation.aggregate_k_hop_features(adj, node_emb, self.query_graph_hop)
            if batch is not None:
                query_emb_pool = global_mean_pool(query_embeddings, batch)
            else:
                query_emb_pool = query_embeddings.mean(dim=0, keepdim=True)

            if ring_mean.shape[1] != self.emb_size:
                ring_mean = ring_mean[:, :self.emb_size]

            ring_feat = self.ring_proj(ring_mean)
            hidden_embedding = torch.cat([query_emb_pool, rag_embedding, ring_feat], dim=-1)

            decode_label = self.decoder(hidden_embedding)
            graph_logits = self.graph_decoder(graph_emb)
            task_logits = decode_label * (1 - self.graph_logit_weight) + graph_logits * self.graph_logit_weight
            label_logits = rag_label * self.fusion_gamma + task_logits * (1 - self.fusion_gamma)

            if return_ring_loss:
                return label_logits, ring_loss, ring_mean
            else:
                return label_logits
        else:
            rag_label = torch.mean(rag_labels, dim=1)
            return rag_label

    def forward_with_loss(self, features, adj, complex_batch, label, batch=None, ptr=None):
        """
        前向传播 + 构建 loss，专用于训练阶段。
        返回: (total_loss, logits, metrics_dict)
        """
        node_emb, _ = self.pretrain_model.embed(features, adj)

        # === Graph-level pooling ===
        if batch is not None:
            graph_emb = global_mean_pool(node_emb, batch)
        elif ptr is not None:
            graph_emb = [node_emb[ptr[i]:ptr[i + 1]].mean(dim=0) for i in range(len(ptr) - 1)]
            graph_emb = torch.stack(graph_emb, dim=0)
        else:
            graph_emb = node_emb.mean(dim=0, keepdim=True)

        # === Ring feature + loss ===
        ring_loss = torch.tensor(0.0, device=node_emb.device)
        ring_means = []
        view1_all, view2_all = [], []
        valid_ring_graphs = 0
        total_rings = 0

        if batch is not None:
            num_graphs = batch.max().item() + 1
            for i in range(num_graphs):
                node_emb_i = node_emb[batch == i]
                complex_obj = complex_batch[i]

                if complex_obj is None or \
                        2 not in complex_obj.cochains or \
                        complex_obj.cochains[2] is None or \
                        complex_obj.cochains[2].boundary_index is None or \
                        complex_obj.cochains[2].boundary_index.size(1) == 0:
                    ring_mean_i = torch.zeros(self.emb_size, device=node_emb.device, requires_grad=True)
                    single_loss = torch.tensor(0.0, device=node_emb.device, requires_grad=True)
                else:
                    view1_list, view2_list, ring_mean_i = ring_views_from_boundary(
                        node_emb_i,
                        complex_obj.cochains[2].boundary_index.to(node_emb.device),
                        self._edge_index_from_complex(complex_obj).to(node_emb.device),
                        noise_std=0.05 if self.training else 0.0
                    )
                    view1_all.extend(view1_list)
                    view2_all.extend(view2_list)
                    valid_ring_graphs += int(len(view1_list) > 0)
                    total_rings += len(view1_list)
                    single_loss = torch.tensor(0.0, device=node_emb.device)
                ring_loss += single_loss
                ring_means.append(ring_mean_i)

            ring_loss = ring_contrastive_loss_from_views(
                view1_all, view2_all, device=node_emb.device
            )
            ring_mean = torch.stack(ring_means, dim=0)
        else:
            # 单图逻辑简化
            single_loss, ring_mean = ring_contrastive_loss(
                node_emb,
                complex_batch.cochains[2].boundary_index.to(node_emb.device),
                self._edge_index_from_complex(complex_batch).to(node_emb.device)
            )
            ring_loss = single_loss
            ring_mean = ring_mean.unsqueeze(0)

        ring_feat = self.ring_proj(ring_mean)
        query_keys = torch.cat([graph_emb, ring_mean], dim=-1)

        rag_embeddings, rag_labels, rag_weights = self.toy_graph_base.retrieve(
            query_keys, adj, complex_batch, ring_mean, add_noise=self.training and self.noise_finetune
        )
        rag_embedding = torch.sum(rag_weights.unsqueeze(-1) * rag_embeddings, dim=1)
        rag_label = torch.sum(rag_weights.unsqueeze(-1) * rag_labels, dim=1)

        query_embeddings = Propagation.aggregate_k_hop_features(adj, node_emb, self.query_graph_hop)
        if batch is not None:
            query_emb_pool = global_mean_pool(query_embeddings, batch)
        else:
            query_emb_pool = query_embeddings.mean(dim=0, keepdim=True)

        decoder_input = torch.cat([query_emb_pool, rag_embedding, ring_feat], dim=-1)
        decode_logits = self.decoder(decoder_input)
        graph_logits = self.graph_decoder(graph_emb)
        task_logits = decode_logits * (1 - self.graph_logit_weight) + graph_logits * self.graph_logit_weight
        label_logits = rag_label * self.fusion_gamma + task_logits * (1 - self.fusion_gamma)

        # 核心修复：计算交叉熵分类损失，并将拓扑环损失按指定的 ring_weight 结合进总损失中
        cls_loss = F.cross_entropy(label_logits, label)
        total_loss = cls_loss + self.ring_weight * ring_loss

        return total_loss, label_logits, {
            "cls_loss": cls_loss,
            "ring_loss": ring_loss,
            "valid_ring_graphs": valid_ring_graphs,
            "total_rings": total_rings
        }

    def get_graph_and_ring_embeddings(self, features, adj, complex_batch, batch=None, ptr=None):
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            node_emb, _ = self.pretrain_model.embed(features, adj)

            if batch is not None:
                graph_emb = global_mean_pool(node_emb, batch)
            elif ptr is not None:
                graph_emb = [node_emb[ptr[i]:ptr[i + 1]].mean(dim=0) for i in range(len(ptr) - 1)]
                graph_emb = torch.stack(graph_emb, dim=0)
            else:
                graph_emb = node_emb.mean(dim=0, keepdim=True)

            ring_means = []
            if batch is not None:
                num_graphs = batch.max().item() + 1
                for i in range(num_graphs):
                    node_emb_i = node_emb[batch == i]
                    complex_obj = complex_batch[i]

                    if (complex_obj is None or
                            2 not in complex_obj.cochains or
                            complex_obj.cochains[2] is None or
                            complex_obj.cochains[2].boundary_index is None or
                            complex_obj.cochains[2].boundary_index.size(1) == 0):
                        ring_mean_i = torch.zeros(self.emb_size, device=device)
                    else:
                        _, ring_mean_i = ring_contrastive_loss(
                            node_emb_i,
                            complex_obj.cochains[2].boundary_index.to(device),
                            self._edge_index_from_complex(complex_obj).to(device)
                        )
                    ring_means.append(ring_mean_i)
                ring_mean = torch.stack(ring_means, dim=0)
            else:
                _, ring_mean = ring_contrastive_loss(
                    node_emb,
                    complex_batch.cochains[2].boundary_index.to(device),
                    self._edge_index_from_complex(complex_batch).to(device)
                )
                ring_mean = ring_mean.unsqueeze(0)

            ring_feat = self.ring_proj(ring_mean)
            query_keys = torch.cat([graph_emb, ring_mean], dim=-1)
            rag_embeddings, rag_labels, rag_weights = self.toy_graph_base.retrieve(
                query_keys, adj, complex_batch, ring_mean, add_noise=False
            )
            rag_embedding = torch.sum(rag_weights.unsqueeze(-1) * rag_embeddings, dim=1)

        return graph_emb.cpu(), ring_feat.cpu(), rag_embedding.cpu()

    def get_graph_embedding_noring(self, features, adj, batch=None, ptr=None):
        self.eval()
        with torch.no_grad():
            node_emb, _ = self.pretrain_model.embed(features, adj)

            if batch is not None:
                graph_emb = global_mean_pool(node_emb, batch)
            elif ptr is not None:
                graph_emb_list = [
                    node_emb[ptr[i]:ptr[i + 1]].mean(dim=0)
                    for i in range(len(ptr) - 1)
                ]
                graph_emb = torch.stack(graph_emb_list, dim=0)
            else:
                graph_emb = node_emb.mean(dim=0, keepdim=True)

            zero_ring = torch.zeros_like(graph_emb)
            query_keys = torch.cat([graph_emb, zero_ring], dim=-1)
            search_ring_feat = torch.zeros_like(graph_emb)

            rag_embeddings, rag_labels, rag_weights = self.toy_graph_base.retrieve(
                query_keys, adj, None, search_ring_feat, False
            )

            attn = rag_weights.unsqueeze(-1)
            rag_embedding = torch.sum(attn * rag_embeddings, dim=1)
            hidden_embedding = torch.cat([graph_emb, rag_embedding], dim=-1)

            return graph_emb, rag_embedding, hidden_embedding

    def encode_graph(self, features, adj, complex_batch=None, batch=None):
        self.eval()
        with torch.no_grad():
            node_emb, _ = self.pretrain_model.embed(features, adj)

            if batch is not None:
                graph_emb = global_mean_pool(node_emb, batch)
            else:
                graph_emb = node_emb.mean(dim=0, keepdim=True)

        return node_emb, graph_emb
