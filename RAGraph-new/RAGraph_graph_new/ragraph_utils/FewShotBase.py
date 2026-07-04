import torch

from .TaskDecoder import TaskDecoder
from .SimilarityFunctions import SimilarityFunctions

class FewShotBase:
    def __init__(self, dataset_name: str, num_classes: int, pretrain_model):

        self.fewshot_adj: torch.Tensor = torch.load(f"data/fewshot_{dataset_name}_graph/testset/adj.pt").cuda()
        self.fewshot_feature: torch.Tensor = torch.load(f"data/fewshot_{dataset_name}_graph/testset/feature.pt").cuda()
        self.fewshot_label: torch.Tensor = torch.load(f"data/fewshot_{dataset_name}_graph/testset/labels.pt").cuda()
        self.fewshot_graph_len = torch.load(f"data/fewshot_{dataset_name}_graph/testset/graph_len.pt").cuda()

        # 如果 adj 是 3D batched (B, N_max, N_max)，需要转为 2D block-diagonal (N_total, N_total)
        # 因为 to_dense_adj 生成的 adj 有 padding，必须用 graph_len 提取有效部分
        if self.fewshot_adj.dim() == 3:
            B = self.fewshot_adj.size(0)
            # graph_len 可能是 node-to-graph 分配向量 (如 batch.batch)，也可能是每图节点数列表
            if self.fewshot_graph_len.numel() == self.fewshot_feature.size(0):
                # node-to-graph assignment -> 计算每图节点数
                graph_sizes = []
                for g in range(B):
                    graph_sizes.append((self.fewshot_graph_len == g).sum().item())
            elif self.fewshot_graph_len.numel() == B:
                graph_sizes = self.fewshot_graph_len.tolist()
            else:
                raise ValueError(
                    f"graph_len 格式不匹配: graph_len.numel()={self.fewshot_graph_len.numel()}, "
                    f"B={B}, feature_nodes={self.fewshot_feature.size(0)}")
            # 提取每个图的有效邻接子矩阵，拼成 block-diagonal
            adj_blocks = []
            for i in range(B):
                n = int(graph_sizes[i])
                adj_blocks.append(self.fewshot_adj[i, :n, :n])
            self.fewshot_adj = torch.block_diag(*adj_blocks)

        self.fewshot_embeddings: torch.Tensor = pretrain_model.inference(self.fewshot_feature, self.fewshot_adj).squeeze()
        self.fewshot_one_hot_label = torch.nn.functional.one_hot(self.fewshot_label.long(), num_classes=num_classes).float().cuda()

    def __call__(self, search_embeddings: torch.Tensor, decoder: TaskDecoder) -> torch.Tensor:
        search_embeddings_decoded = decoder(search_embeddings)
        fewshot_embeddings_decoded = decoder(self.fewshot_embeddings)

        # (batch_size, fewshot_num)
        similarity = SimilarityFunctions.calculate_cosine_similarity(search_embeddings_decoded, fewshot_embeddings_decoded)

        # (batch_size, num_classes) = (batch_size, fewshot_num) * (fewshot_num, num_classes)
        predict_label = torch.matmul(similarity, self.fewshot_one_hot_label)

        return predict_label
