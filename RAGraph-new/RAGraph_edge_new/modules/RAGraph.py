import networkx as nx
import torch
import torch.nn as nn
from torch import Tensor
from modules.base_model import BaseModel
from utils.parse_args import args
import torch.nn.functional as F
from modules.utils import EdgelistDrop
import logging
from modules.utils import scatter_add, scatter_sum
from torch_scatter import scatter_softmax
from .complex import Cochain
from modules.extract_ring import extract_ring_mean_batch
from modules.ragraph_utils.Augmentation import Augmentation
from modules.ragraph_utils.InverseSampling import InverseSampling
from modules.ragraph_utils.SimilarityFunctions import SimilarityFunctions

init = nn.init.xavier_uniform_
logger = logging.getLogger('train_logger')

class RAGraph(BaseModel):
    def __init__(self, dataset, pretrained_model=None, phase='pretrain', use_RAG=True, use_noise=False, use_LoRA=True, LoRA_rank=16):
        super().__init__(dataset)
        self.dataset = dataset  # 保存 dataset 方便后面使用
        self.adj = self._make_binorm_adj(dataset.graph)
        self.edges = self.adj._indices().t()
        self.edge_norm = self.adj._values()
        self.edge_times = [dataset.edge_time_dict[e[0]][e[1]] for e in self.edges.cpu().tolist()]
        self.edge_times = torch.LongTensor(self.edge_times).to(args.device)

        self.phase = phase
        self.use_RAG = use_RAG
        self.use_noise = use_noise and phase == 'finetune'

        # 链路预测 MLP
        self.link_predictor = nn.Sequential(
            nn.Linear(args.emb_size * 3, args.emb_size),
            nn.ReLU(),
            nn.Linear(args.emb_size, 1)
        )

        # === 数据集相关参数 ===
        data_path = args.data_path
        self.retrieve_weight = 0.3
        self.ring_proj = nn.Linear(self.emb_size, self.emb_size)
        self.ring_weight = 0.01  # 先给个小权重，可以调
        self.num_inverse_sample = getattr(args, "num_inverse_sample", 0)  # 新增
        if 'amazon' in data_path:
            self.batch_size, self.retrieve_num = (32768, 50) if self.phase == 'vanilla' else (4096, 10)
        elif 'koubei' in data_path or 'taobao' in data_path:
            self.batch_size, self.retrieve_num = (512, 100000) if self.phase == 'vanilla' else (4096, 20)
        else:
            raise NotImplementedError

        self.resource_graph_radius = args.num_layers
        self.resource_keys = None
        self.resource_times = None
        self.resource_values = None

        # === Embedding 初始化 ===
        if self.phase in ['pretrain', 'for_tune']:
            self.user_embedding = nn.Parameter(init(torch.empty(self.num_users, self.emb_size)))
            self.item_embedding = nn.Parameter(init(torch.empty(self.num_items, self.emb_size)))
            self.emb_gate = lambda x: x
        elif self.phase == 'vanilla':
            pre_user_emb, pre_item_emb = pretrained_model.generate(return_ring=False)
            self.user_embedding = nn.Parameter(pre_user_emb).requires_grad_(False)
            self.item_embedding = nn.Parameter(pre_item_emb).requires_grad_(False)
            if self.use_RAG:
                self._make_resource_graph(pretrained_model)
            self.emb_gate = lambda x: x
        elif self.phase == 'finetune':
            pre_user_emb, pre_item_emb = pretrained_model.generate(return_ring=False)
            if self.use_RAG:
                self._make_resource_graph(pretrained_model)
            self.user_embedding = nn.Parameter(pre_user_emb).requires_grad_(True)
            self.item_embedding = nn.Parameter(pre_item_emb).requires_grad_(True)
            self.gating_weight = nn.Parameter(init(torch.empty(args.emb_size, args.emb_size)))
            self.gating_bias = nn.Parameter(init(torch.empty(1, args.emb_size)))
            self.emb_dropout = nn.Dropout(args.emb_dropout)
            self.emb_gate = lambda x: self.emb_dropout(
                torch.mul(x, torch.sigmoid(torch.matmul(x, self.gating_weight) + self.gating_bias))
            )
        self.ring_mlp = nn.Sequential(
            nn.Linear(self.emb_size * 2, self.emb_size),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size)
        )

        self.edge_dropout = EdgelistDrop()
        logger.info(f"Max Time Step: {self.edge_times.max()}")

    def _make_resource_graph(self, pretrained_model: BaseModel):
        pre_user_emb, pre_item_emb = pretrained_model.generate(return_ring=False)
        all_emb = torch.cat([pre_user_emb, pre_item_emb], dim=0)
        res_emb = [all_emb]
        for _ in range(self.resource_graph_radius):
            all_emb = self._agg(res_emb[-1], self.edges, self.edge_norm)
            res_emb.append(all_emb)
        dual_res_emb = res_emb[0::2]
        all_logits = sum(dual_res_emb)
        sample_prob = InverseSampling.compute_sample_prob(self.adj)
        num_loop = 1
        for i in range(num_loop):
            aug_keys, aug_values = all_emb, all_logits
            if self.num_inverse_sample > 0:
                sample_mask = torch.multinomial(sample_prob, num_samples=self.num_inverse_sample, replacement=True)
                sample_keys, sample_values = aug_keys[sample_mask], aug_values[sample_mask]
            else:
                sample_keys, sample_values = aug_keys, aug_values
            self.resource_keys = torch.cat((self.resource_keys, sample_keys), dim=0) if self.resource_keys is not None else sample_keys
            self.resource_values = torch.cat((self.resource_values, sample_values), dim=0) if self.resource_values is not None else sample_values

    def _agg(self, all_emb, edges, edge_norm):
        src_emb = all_emb[edges[:, 0]] * edge_norm.unsqueeze(1)
        dst_emb = scatter_sum(src_emb, edges[:, 1], dim=0, dim_size=self.num_users+self.num_items)
        return dst_emb

    def _relative_edge_time_encoding(self, edges, edge_times, max_step=None):
        edge_times = edge_times.float()
        max_step = edge_times.max() if max_step is None else max_step
        edge_times = (edge_times - edge_times.min()) / (max_step - edge_times.min())
        dst_nodes = edges[:, 1]
        time_norm = scatter_softmax(edge_times, dst_nodes, dim_size=self.num_users+self.num_items)
        return time_norm

    def extract_pair_ring_feature(self, node_emb, complex_batch, batch_vector, edge_index):
        pair_ring_feats = []
        for u, v in edge_index:
            graph_idx = batch_vector[u]
            complex_obj = complex_batch[graph_idx]
            if complex_obj is None or 2 not in complex_obj.cochains:
                pair_ring_feats.append(torch.zeros(node_emb.size(1), device=node_emb.device))
                continue
            cochain2 = complex_obj.cochains[2]
            cochain1 = complex_obj.cochains[1]
            if cochain2 is None or cochain2.boundary_index is None or cochain2.boundary_index.size(1) == 0:
                pair_ring_feats.append(torch.zeros(node_emb.size(1), device=node_emb.device))
                continue
            ring_dict = {}
            for i in range(cochain2.boundary_index.size(1)):
                edge_id = cochain2.boundary_index[0, i].item()
                if edge_id >= cochain1.upper_index.size(1):  # 修复变量名
                    continue
                u1 = cochain1.upper_index[0, edge_id].item()
                v1 = cochain1.upper_index[1, edge_id].item()
                ring_id = cochain2.boundary_index[1, i].item()
                ring_dict.setdefault(ring_id, set()).update([u1, v1])
            common_rings = [rid for rid, nodes in ring_dict.items() if u.item() in nodes and v.item() in nodes]
            if common_rings:
                emb_list = [node_emb[list(ring_dict[rid])].mean(dim=0) for rid in common_rings]
                pair_ring_feats.append(torch.stack(emb_list).mean(dim=0))
            else:
                pair_ring_feats.append(torch.zeros(node_emb.size(1), device=node_emb.device))
        return torch.stack(pair_ring_feats, dim=0)

    def forward(self, edges, edge_norm, edge_times, max_time_step=None, complex_batch=None, batch_vector=None,
                return_all=True):
        # === 构建 ComplexBatch ===
        if complex_batch is None or batch_vector is None:
            # print("[DEBUG] Forward: Auto-building ComplexBatch...")
            complex_batch, batch_vector, global_node_idx = self.dataset.build_complex_batch()
        else:
            # print(f"[DEBUG] Forward: Received ComplexBatch externally, size={len(complex_batch)}")
            global_node_idx = None

        # === 时间编码 ===
        time_norm = self._relative_edge_time_encoding(edges, edge_times, max_step=max_time_step)
        edge_norm = edge_norm * 0.5 + time_norm * 0.5

        # === 聚合节点特征 ===
        all_emb = torch.cat([self.user_embedding, self.item_embedding], dim=0)
        all_emb = self.emb_gate(all_emb)
        res_emb = [all_emb]
        for _ in range(args.num_layers):
            all_emb = self._agg(res_emb[-1], edges, edge_norm)
            res_emb.append(all_emb)
        res_emb = sum(res_emb)

        # === 子图 & 环特征 ===
        if global_node_idx is not None:
            subgraph_emb = all_emb[global_node_idx]
        else:
            subgraph_emb = all_emb
        ring_mean = extract_ring_mean_batch(subgraph_emb, complex_batch, batch_vector)  # [num_subgraphs, emb]

        # === 返回 user/item + 环特征 ===
        user_res_emb, item_res_emb = res_emb.split([self.num_users, self.num_items], dim=0)
        if return_all:
            return user_res_emb, item_res_emb, all_emb, ring_mean
        else:
            return user_res_emb, item_res_emb

    def cal_loss(self, batch_data, complex_batch=None, batch_vector=None, mode="bpr"):
        edges, dropout_mask = self.edge_dropout(self.edges, 1 - args.edge_dropout, return_mask=True)
        edge_norm = self.edge_norm[dropout_mask]
        edge_times = self.edge_times[dropout_mask]
        if mode == "bpr":
            users, pos_items, neg_items = batch_data
            user_emb, item_emb = self.forward(edges, edge_norm, edge_times, complex_batch=complex_batch, batch_vector=batch_vector,return_all=False)
            batch_user_emb, pos_item_emb, neg_item_emb = user_emb[users], item_emb[pos_items], item_emb[neg_items]
            rec_loss = self._bpr_loss(batch_user_emb, pos_item_emb, neg_item_emb)
            reg_loss = args.weight_decay * self._reg_loss(users, pos_items, neg_items)
            return rec_loss + reg_loss, {"rec_loss": rec_loss.item(), "reg_loss": reg_loss.item()}
        elif mode == "bce":
            users, items, labels = batch_data
            user_emb, item_emb, all_emb = self.forward(edges, edge_norm, edge_times, complex_batch=complex_batch, batch_vector=batch_vector, return_all=True)
            edge_index = torch.stack([users, items], dim=1)
            pair_ring_feats = self.extract_pair_ring_feature(all_emb, complex_batch, batch_vector, edge_index)
            edge_input = torch.cat([user_emb[users], item_emb[items], pair_ring_feats], dim=1)
            logits = self.link_predictor(edge_input).squeeze()
            bce_loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            reg_loss = args.weight_decay * (user_emb[users].norm(2).pow(2) + item_emb[items].norm(2).pow(2)) / len(users)
            return bce_loss + reg_loss, {"bce_loss": bce_loss.item(), "reg_loss": reg_loss.item()}

    @torch.no_grad()
    def generate(self, max_time_step=None, return_ring=False):
        user_emb, item_emb, _, ring_feat = self.forward(
            self.edges, self.edge_norm, self.edge_times,
            max_time_step=max_time_step,
            return_all=True
        )
        if return_ring:
            return user_emb, item_emb, ring_feat
        else:
            return user_emb, item_emb

    @torch.no_grad()
    def rating(self, user_emb, item_emb, ring_feat=None):
        scores = torch.matmul(user_emb, item_emb.t())
        if ring_feat is not None:
            ring_bias = self.ring_proj(ring_feat).mean(dim=0, keepdim=True).repeat(item_emb.size(0), 1)
            # 融合 item_emb 和 ring_feat
            enhanced_item = self.ring_mlp(torch.cat([item_emb, ring_bias], dim=1))
            scores = torch.matmul(user_emb, enhanced_item.t())
        return scores

    def _reg_loss(self, users, pos_items, neg_items):
        u_emb = self.user_embedding[users]
        pos_i_emb = self.item_embedding[pos_items]
        neg_i_emb = self.item_embedding[neg_items]
        reg_loss = (1/2)*(u_emb.norm(2).pow(2) + pos_i_emb.norm(2).pow(2) + neg_i_emb.norm(2).pow(2))/float(len(users))
        return reg_loss
