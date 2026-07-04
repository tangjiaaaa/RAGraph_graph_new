from .TaskDecoder import TaskDecoder
from .FewShotBase import FewShotBase
from .ToyGraphBase import ToyGraphBase
from .Propagation import Propagation
from .SimilarityFunctions import SimilarityFunctions
from .utility2 import seed_everything, process_tu_dataset
from .complex import Cochain,Complex,ComplexBatch
from .extract_ring import (
    ring_contrastive_loss,
    extract_ring_mean,
    ring_contrastive_loss_new,
    ring_views_from_boundary,
    ring_contrastive_loss_from_views,
)
from .DiffLiftRingSelector import DiffLiftRingSelector
from .TaskAwareRetriever import TaskAwareReranker, retrieval_alignment_loss
import torch
def build_complex_from_graph(data):
    """
    给单个 PyG Data 图构建 Complex，包含0/1/2维胞腔
    """
    num_nodes = data.num_nodes
    edge_index = data.edge_index

    # 2-cell (环) 边界关系
    # 你以前 FCB 找环的接口
    from .extract_ring import extract_ring_mean
    ring_boundary_index = extract_ring_mean(edge_index, num_nodes)

    # 构造 Cochain
    cochains = {
        0: Cochain(indices=torch.arange(num_nodes)),
        1: Cochain(boundary_index=edge_index),
        2: Cochain(boundary_index=ring_boundary_index)
    }
    return Complex(cochains)
def fewshot_mean_logits(logits: torch.Tensor, labels: torch.Tensor, num_classes: int):
    """
    计算每个类别的 support 样本 logits 平均值
    logits: [N, C]
    labels: [N]
    num_classes: 类别数
    return: [num_classes, C] 类原型
    """
    class_means = []
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() == 0:
            class_means.append(torch.zeros(logits.size(1)))
        else:
            class_means.append(logits[mask].mean(dim=0))
    return torch.stack(class_means, dim=0)

def fewshot_predict_labels_by_mean(logits: torch.Tensor, mean_logits: torch.Tensor):
    """
    用 support 类原型做最近类别匹配
    logits: [N, C]
    mean_logits: [num_classes, C]
    return: 预测类别 [N]
    """
    # 计算每个 query 与各类别原型的欧式距离（或内积）
    distances = torch.cdist(logits.cpu(), mean_logits.cpu(), p=2)  # 欧式距离
    preds = torch.argmin(distances, dim=1)
    return preds
