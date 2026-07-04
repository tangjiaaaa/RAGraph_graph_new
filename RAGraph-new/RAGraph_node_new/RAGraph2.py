import torch
import torch.nn as nn
import scipy.sparse as sp
# from torch_geometric.loader import DataLoader
# from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader

from pretrain3 import ring_contrastive_loss
from utils import process
from ragraph_utils import ToyGraphBase, Propagation, TaskDecoder


class RAGraph(nn.Module):
    def __init__(self, pretrain_model, resource_dataset, feture_size, num_class, emb_size, finetune=True, noise_finetune=False) -> None:
        super(RAGraph, self).__init__()

        self.emb_size = emb_size
        self.num_class = num_class
        self.pretrain_model = pretrain_model

        self.retrieve_weight = 0.5
        self.label_weight = 0.5
        self.finetune = finetune

        self.noise_finetune = noise_finetune
        if self.noise_finetune:
            assert self.finetune

        self.query_graph_hop = 3
        self.toy_graph_base = ToyGraphBase(pretrain_model, num_class, emb_size, self.query_graph_hop)
        self.toy_graph_base.build_toy_graph(resource_dataset)

        if self.finetune:
            self.decoder = TaskDecoder(emb_size, emb_size, num_class)
            self.reset_parameters()

        self.toy_graph_base.show()

    def reset_parameters(self):
        self.decoder.reset_parameters()

    def forward(self, features, adj, complex_obj=None, return_ring_loss=False):

        pretrain_embedddings = self.pretrain_model.inference(features, adj)
        pretrain_embedddings = torch.nan_to_num(pretrain_embedddings, nan=0.0, posinf=0.0, neginf=0.0)

        add_noise = self.training and self.noise_finetune
        rag_embeddings, rag_labels = self.toy_graph_base.retrieve(pretrain_embedddings, adj, add_noise)
        rag_embeddings = torch.nan_to_num(rag_embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        rag_labels = torch.nan_to_num(rag_labels, nan=0.0, posinf=0.0, neginf=0.0)

        if self.finetune:
            rag_label = torch.mean(rag_labels, dim=1)
            rag_embedding = torch.sum(rag_embeddings, dim=1)

            query_embeddings = Propagation.aggregate_k_hop_features(adj, pretrain_embedddings, self.query_graph_hop)

            hidden_embedding = query_embeddings * (1 - self.retrieve_weight) + rag_embedding * self.retrieve_weight
            decode_label = self.decoder(hidden_embedding)
            decode_label = torch.softmax(decode_label, dim=1)

            label_logits = decode_label * (1 - self.label_weight) + rag_label * self.label_weight
            label_logits = torch.nan_to_num(label_logits, nan=0.0, posinf=0.0, neginf=0.0)

            if return_ring_loss:
                ring_loss = self.compute_ring_contrastive_loss(pretrain_embedddings, complex_obj)
                return label_logits, ring_loss

            return label_logits
        else:
            rag_label = torch.mean(rag_labels, dim=1)
            return rag_label

    def compute_ring_contrastive_loss(self, features, complex_obj):
        """
        This function will compute the contrastive loss based on the ring structure in complex_obj.
        """
        # Assuming complex_obj contains cochains for ring boundaries and edges
        if (
            complex_obj is not None
            and hasattr(complex_obj, "cochains")
            and 2 in complex_obj.cochains
            and complex_obj.cochains[2] is not None
            and complex_obj.cochains[2].boundary_index is not None
            and complex_obj.cochains[2].boundary_index.numel() > 0
        ):
            ring_boundary = complex_obj.cochains[2].boundary_index.cuda()
            edge_index = complex_obj.cochains[1].boundary_index.cuda()
            return ring_contrastive_loss(features, ring_boundary, edge_index)
        else:
            return torch.tensor(0.0, requires_grad=True).cuda()

