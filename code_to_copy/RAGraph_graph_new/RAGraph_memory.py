import torch
import torch.nn as nn
import torch.nn.functional as F
from ragraph_utils import (
    ToyGraphBase,
    Propagation,
    FewShotBase,
    TaskDecoder,
    ring_contrastive_loss_from_views,
    ring_views_from_boundary,
    DiffLiftRingSelector,
    TaskAwareReranker,
    retrieval_alignment_loss,
)
from torch_geometric.nn import global_mean_pool


class RAGraph(nn.Module):
    """ReTAG memory variant integrated with the existing RAGraph_graph_new API.

    This model keeps the original RAGraph2 data flow:
    features, adj, complex_batch, batch/ptr -> logits or forward_with_loss.

    Added components:
    - DiffLiftRingSelector: learns task-adaptive weights over existing 2-cells.
    - TaskAwareReranker: reranks retrieved cellular memories.
    - memory utility update: calibrates memory usefulness during training only.
    """

    def __init__(
        self,
        pretrain_model,
        resource_dataset,
        feture_size,
        num_class,
        emb_size,
        finetune=True,
        noise_finetune=False,
        dataset_name=None,
        ring_weight=0.05,
        retrieval_weight=0.1,
        query_graph_hop=2,
        retrieve_num=5,
        fusion_gamma=0.2,
        max_ring=10,
        use_diff_lifting=True,
        use_task_rerank=True,
        use_memory_reflection=True,
        memory_utility_weight=0.1,
    ) -> None:
        super(RAGraph, self).__init__()
        self.emb_size = emb_size
        self.num_class = num_class
        self.pretrain_model = pretrain_model
        self.ring_weight = ring_weight
        self.retrieval_weight = retrieval_weight
        self.fusion_gamma = fusion_gamma
        self.dataset_name = dataset_name
        self.finetune = finetune
        self.noise_finetune = noise_finetune
        self.query_graph_hop = query_graph_hop
        self.use_diff_lifting = use_diff_lifting
        self.use_task_rerank = use_task_rerank
        self.use_memory_reflection = use_memory_reflection

        if self.noise_finetune:
            assert self.finetune

        self.ring_proj = nn.Sequential(
            nn.Linear(emb_size, emb_size * 2),
            nn.ReLU(),
            nn.Linear(emb_size * 2, emb_size),
        )
        self.ring_selector = DiffLiftRingSelector(emb_size)
        self.reranker = TaskAwareReranker(
            query_dim=2 * emb_size,
            candidate_dim=2 * emb_size,
            hidden_dim=2 * emb_size,
        )

        self.toy_graph_base = ToyGraphBase(pretrain_model, num_class, emb_size, self.query_graph_hop, max_ring=max_ring)
        self.toy_graph_base.retrieve_num = retrieve_num
        self.toy_graph_base.utility_weight = memory_utility_weight if use_memory_reflection else 0.0
        self.toy_graph_base.build_toy_graph(resource_dataset)
        self.toy_graph_base.noise_std = 0.05

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
    def _edge_boundary_from_complex(complex_obj):
        edge_cochain = complex_obj.cochains[1]
        return getattr(edge_cochain, "edge_index", edge_cochain.boundary_index)

    def _static_ring_mean(self, node_emb, complex_obj):
        view1_list, view2_list, ring_mean = ring_views_from_boundary(
            node_emb,
            complex_obj.cochains[2].boundary_index.to(node_emb.device),
            self._edge_boundary_from_complex(complex_obj).to(node_emb.device),
            noise_std=0.05 if self.training else 0.0,
        )
        ring_loss = ring_contrastive_loss_from_views(view1_list, view2_list, device=node_emb.device)
        return ring_mean, ring_loss, {
            "ring_count": len(view1_list),
            "selected_mass": torch.tensor(float(len(view1_list)), device=node_emb.device),
        }

    def _learned_ring_mean(self, node_emb, complex_obj):
        ring_mean, selector_loss, info = self.ring_selector(
            node_emb,
            complex_obj.cochains[2].boundary_index.to(node_emb.device),
            self._edge_boundary_from_complex(complex_obj).to(node_emb.device),
        )
        view1_list, view2_list, _ = ring_views_from_boundary(
            node_emb,
            complex_obj.cochains[2].boundary_index.to(node_emb.device),
            self._edge_boundary_from_complex(complex_obj).to(node_emb.device),
            noise_std=0.05 if self.training else 0.0,
        )
        contrastive_loss = ring_contrastive_loss_from_views(view1_list, view2_list, device=node_emb.device)
        return ring_mean, contrastive_loss + selector_loss, info

    def _compute_ring_batch(self, node_emb, complex_batch, batch=None):
        ring_means = []
        total_ring_loss = torch.tensor(0.0, device=node_emb.device)
        valid_ring_graphs = 0
        total_rings = 0
        selected_mass = torch.tensor(0.0, device=node_emb.device)

        if batch is not None:
            num_graphs = batch.max().item() + 1
            for i in range(num_graphs):
                node_emb_i = node_emb[batch == i]
                complex_obj = complex_batch[i] if complex_batch is not None else None
                ring_mean_i, ring_loss_i, info = self._compute_single_ring(node_emb_i, complex_obj)
                ring_means.append(ring_mean_i)
                total_ring_loss = total_ring_loss + ring_loss_i
                valid_ring_graphs += int(info["ring_count"] > 0)
                total_rings += int(info["ring_count"])
                selected_mass = selected_mass + info["selected_mass"]
            ring_mean = torch.stack(ring_means, dim=0)
            total_ring_loss = total_ring_loss / max(num_graphs, 1)
        else:
            ring_mean, total_ring_loss, info = self._compute_single_ring(node_emb, complex_batch)
            ring_mean = ring_mean.unsqueeze(0)
            valid_ring_graphs = int(info["ring_count"] > 0)
            total_rings = int(info["ring_count"])
            selected_mass = info["selected_mass"]

        return ring_mean, total_ring_loss, {
            "valid_ring_graphs": valid_ring_graphs,
            "total_rings": total_rings,
            "selected_mass": selected_mass.detach(),
        }

    def _compute_single_ring(self, node_emb, complex_obj):
        if (
            complex_obj is None
            or 2 not in complex_obj.cochains
            or complex_obj.cochains[2] is None
            or complex_obj.cochains[2].boundary_index is None
            or complex_obj.cochains[2].boundary_index.size(1) == 0
        ):
            zero = torch.zeros(self.emb_size, device=node_emb.device)
            return zero, zero.sum(), {"ring_count": 0, "selected_mass": zero.sum()}

        if self.use_diff_lifting:
            return self._learned_ring_mean(node_emb, complex_obj)
        return self._static_ring_mean(node_emb, complex_obj)

    def _pool_graph(self, node_emb, batch=None, ptr=None):
        if batch is not None:
            return global_mean_pool(node_emb, batch)
        if ptr is not None:
            graph_emb = [node_emb[ptr[i]:ptr[i + 1]].mean(dim=0) for i in range(len(ptr) - 1)]
            return torch.stack(graph_emb, dim=0)
        return node_emb.mean(dim=0, keepdim=True)

    def _decode(self, node_emb, graph_emb, adj, ring_mean, rag_embeddings, rag_labels, rag_weights, batch=None):
        rag_embedding = torch.sum(rag_weights.unsqueeze(-1) * rag_embeddings, dim=1)
        rag_label = torch.sum(rag_weights.unsqueeze(-1) * rag_labels, dim=1)

        query_embeddings = Propagation.aggregate_k_hop_features(adj, node_emb, self.query_graph_hop)
        query_emb_pool = global_mean_pool(query_embeddings, batch) if batch is not None else query_embeddings.mean(dim=0, keepdim=True)

        ring_feat = self.ring_proj(ring_mean[:, : self.emb_size])
        hidden_embedding = torch.cat([query_emb_pool, rag_embedding, ring_feat], dim=-1)
        decode_logits = self.decoder(hidden_embedding)
        graph_logits = self.graph_decoder(graph_emb)
        task_logits = decode_logits * (1 - self.graph_logit_weight) + graph_logits * self.graph_logit_weight
        return rag_label * self.fusion_gamma + task_logits * (1 - self.fusion_gamma)

    def forward(self, features, adj, complex_batch=None, batch=None, ptr=None, return_ring_loss=False):
        node_emb, _ = self.pretrain_model.embed(features, adj)
        graph_emb = self._pool_graph(node_emb, batch=batch, ptr=ptr)
        ring_mean, ring_loss, _ = self._compute_ring_batch(node_emb, complex_batch, batch=batch)
        query_keys = torch.cat([graph_emb, ring_mean], dim=-1)

        retrieved = self.toy_graph_base.retrieve(
            query_keys,
            adj,
            complex_batch,
            ring_mean,
            add_noise=self.training and self.noise_finetune,
            return_indices=False,
        )
        rag_embeddings, rag_labels, rag_weights = retrieved

        if self.use_task_rerank:
            rag_weights, _ = self.reranker(query_keys, rag_embeddings, rag_weights)

        if not self.finetune:
            return torch.sum(rag_weights.unsqueeze(-1) * rag_labels, dim=1)

        logits = self._decode(node_emb, graph_emb, adj, ring_mean, rag_embeddings, rag_labels, rag_weights, batch=batch)
        if return_ring_loss:
            return logits, ring_loss, ring_mean
        return logits

    def forward_with_loss(self, features, adj, complex_batch, label, batch=None, ptr=None):
        node_emb, _ = self.pretrain_model.embed(features, adj)
        graph_emb = self._pool_graph(node_emb, batch=batch, ptr=ptr)
        ring_mean, ring_loss, ring_stats = self._compute_ring_batch(node_emb, complex_batch, batch=batch)
        query_keys = torch.cat([graph_emb, ring_mean], dim=-1)

        rag_embeddings, rag_labels, rag_weights, topk_indices = self.toy_graph_base.retrieve(
            query_keys,
            adj,
            complex_batch,
            ring_mean,
            add_noise=self.training and self.noise_finetune,
            return_indices=True,
        )

        retrieval_loss = torch.tensor(0.0, device=node_emb.device)
        if self.use_task_rerank:
            rag_weights, utility_logits = self.reranker(query_keys, rag_embeddings, rag_weights)
            retrieval_loss = retrieval_alignment_loss(utility_logits, rag_labels, label)

        logits = self._decode(node_emb, graph_emb, adj, ring_mean, rag_embeddings, rag_labels, rag_weights, batch=batch)
        cls_loss = F.cross_entropy(logits, label)
        total_loss = cls_loss + self.ring_weight * ring_loss + self.retrieval_weight * retrieval_loss

        if self.training and self.use_memory_reflection:
            self.toy_graph_base.update_memory_utility(topk_indices, label, rag_weights)

        metrics = {
            "cls_loss": cls_loss,
            "ring_loss": ring_loss,
            "retrieval_loss": retrieval_loss,
            "valid_ring_graphs": ring_stats["valid_ring_graphs"],
            "total_rings": ring_stats["total_rings"],
            "selected_mass": ring_stats["selected_mass"],
        }
        return total_loss, logits, metrics
