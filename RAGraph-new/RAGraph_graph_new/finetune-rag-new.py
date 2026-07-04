# run+log.py
import os
import json
import torch
import numpy as np
import argparse
import datetime
import logging
from collections import defaultdict
import random
import time

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch_geometric.data import DataLoader, Dataset
from torch_geometric.datasets import TUDataset
from preprompt import PrePrompt
from RAGraph import RAGraph as LegacyRAGraph
from RAGraph2 import RAGraph as ReTAGRAGraph
from RAGraph_memory import RAGraph as MemoryRAGraph
from ragraph_utils import seed_everything, process_tu_dataset
from sklearn.manifold import TSNE
import torch.nn as nn


# ------------------------- 日志配置 -------------------------
def setup_logger(log_dir="logs", dataset_name="all"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"run_{dataset_name}_{timestamp}.log")

    logger = logging.getLogger("RAGraph")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(fmt='[%(asctime)s] %(levelname)s - %(message)s', datefmt='%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# ------------------------- Few-shot 采样函数 -------------------------
def sample_fewshot_by_class(dataset, num_classes, k=5, seed=42):
    random.seed(seed)
    label_to_indices = defaultdict(list)
    for idx, data in enumerate(dataset):
        label = int(data.y.item()) if hasattr(data.y, 'item') else int(data.y)
        label_to_indices[label].append(idx)

    selected_indices = []
    for c in range(num_classes):
        if len(label_to_indices[c]) < k:
            raise ValueError(f"类 {c} 样本不足 {k} 个，当前仅有 {len(label_to_indices[c])}")
        selected = random.sample(label_to_indices[c], k)
        selected_indices.extend(selected)
    return [dataset[i] for i in selected_indices]


# ------------------------- 投影与加载 -------------------------
def attach_input_projection_to_pretrain(pretrain_model, feature_size, logger):
    expected_in = None
    for name, p in pretrain_model.named_parameters():
        if p.dim() == 2:
            expected_in = p.shape[1]
            break
    if expected_in is None:
        logger.warning("无法从 pretrain_model 推断期望输入维，默认不插入投影层。")
        return

    logger.info(f"Pretrain inferred expected input dim = {expected_in}, dataset feature_size = {feature_size}")

    pretrain_model._input_proj_dict = {}
    original_embed = pretrain_model.embed

    def embed_with_proj(features, adj, *args, **kwargs):
        B = N = D = None
        if features.dim() == 3:
            B, N, D = features.shape
            features_flat = features.view(-1, D)
        else:
            features_flat = features
            D = features.shape[1]

        if D != expected_in:
            if D not in pretrain_model._input_proj_dict:
                proj = nn.Linear(D, expected_in).to(next(pretrain_model.parameters()).device)
                pretrain_model._input_proj_dict[D] = proj
                logger.info(f"创建动态 input_proj: {D} -> {expected_in}")
            features_flat = pretrain_model._input_proj_dict[D](features_flat)

        if features.dim() == 3:
            features = features_flat.view(B, N, -1)
        else:
            features = features_flat
        return original_embed(features, adj, *args, **kwargs)

    pretrain_model.embed = embed_with_proj
    logger.info("已为 pretrain_model 添加动态 input_proj 层。")


def load_partial_state_dict(model, ckpt_path, logger, map_location='cpu', verbose=True):
    raw = torch.load(ckpt_path, map_location=map_location)
    if isinstance(raw, dict) and 'state_dict' in raw and isinstance(raw['state_dict'], dict):
        raw = raw['state_dict']
    if not isinstance(raw, dict):
        raise RuntimeError("Checkpoint 格式不符合预期")
    ckpt = {k.replace('module.', ''): v for k, v in raw.items()}
    model_state = model.state_dict()
    loaded_keys = []
    for k, v in ckpt.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
            loaded_keys.append(k)
    model.load_state_dict(model_state)
    if verbose:
        logger.info(f"[CKPT LOAD] loaded {len(loaded_keys)} params.")


class ListDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list
        self.num_node_attributes = data_list[0].x.size(1) if len(data_list) > 0 else 0

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


# ------------------------- 主流程 -------------------------
def main():
    parser = argparse.ArgumentParser(description="RAGraph Few-shot Finetune")
    parser.add_argument('--dataset', type=str, default='all', help='具体数据集名字，比如 COX2，或者写 all 跑全部')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size')
    parser.add_argument('--epochs', type=int, default=200, help='Downstream Epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--ring_weight', type=float, default=0.1, help='CTCL loss weight lambda')
    parser.add_argument('--test_times', type=int, default=5, help='Run times for variance')
    parser.add_argument('--mode', type=str, default='fewshot', choices=['fewshot', 'full', 'both'],
                        help='Supervision mode')
    parser.add_argument('--finetune_on', type=str, default='val', choices=['train', 'val'],
                        help='Which split is used for downstream finetuning')
    parser.add_argument('--resource_on', type=str, default='val', choices=['train', 'val'],
                        help='Disjoint resource split for retrieval')
    parser.add_argument('--query_hop', type=int, default=2, help='MTMP first-stage hop L')
    parser.add_argument('--retrieve_num', type=int, default=5, help='Top-K retrieved cellular complexes')
    parser.add_argument('--fusion_gamma', type=float, default=0.0, help='Weight for retrieved output o_c')
    parser.add_argument('--save_metric', type=str, default='cls_loss', choices=['loss', 'cls_loss'],
                        help='Metric used to save the best checkpoint')
    parser.add_argument('--max_ring', type=int, default=10, help='Maximum fundamental cycle length for 2-cells')
    parser.add_argument('--shots', type=str, default='5', help='Comma-separated few-shot settings, e.g. 1,2,3,4,5')
    parser.add_argument('--model_variant', type=str, default='legacy', choices=['legacy', 'retag', 'memory'],
                        help='legacy matches RAGraph.py; retag uses RAGraph2.py; memory uses RAGraph_memory.py')
    parser.add_argument('--use_task_rerank', action='store_true',
                        help='Enable task-aware reranking for memory variant')
    parser.add_argument('--use_diff_lifting', action='store_true',
                        help='Enable differentiable ring lifting for memory variant')
    parser.add_argument('--use_memory_reflection', action='store_true',
                        help='Enable memory utility reflection for memory variant')
    args = parser.parse_args()

    # 初始化日志
    logger = setup_logger("logs", args.dataset)
    logger.info("=" * 60)
    logger.info(f"启动实验 | 参数: {vars(args)}")
    logger.info("=" * 60)

    # 确定跑哪些数据集
    if args.dataset.lower() == 'all':
        datasets = ["PROTEINS", "COX2", "BZR", "ENZYMES"]
    else:
        datasets = [args.dataset]

    # 确定跑哪种模式
    supervision_modes = []
    if args.mode in ['fewshot', 'both']: supervision_modes.append(False)
    if args.mode in ['full', 'both']: supervision_modes.append(True)

    shot_k_list = [int(x.strip()) for x in args.shots.split(',') if x.strip()]
    seed_base = 3407

    for use_full_supervision in supervision_modes:
        for dataset_name in datasets:
            current_shot_list = [0] if use_full_supervision else shot_k_list

            for shot_k in current_shot_list:
                mode_name = "Full Supervision" if use_full_supervision else f"{shot_k}-Shot"
                logger.info(f"\n>>>>>>>> 正在启动: {dataset_name} | {mode_name} <<<<<<<<")

                # 纯净加载数据集
                dataset = TUDataset(root='data', name=dataset_name, use_node_attr=True)
                feature_size = dataset.num_node_attributes
                num_classes = dataset.num_classes
                logger.info(
                    f"[DATA] {dataset_name}: feature_size={feature_size}, num_classes={num_classes}, total={len(dataset)}")

                # 初始化预训练模型
                pretrain_model = PrePrompt(feature_size, 256, 'prelu', 1, 0.3, use_proj=True)
                ckpt_path = f'modelset/model_{dataset_name}.pkl'

                if os.path.exists(ckpt_path):
                    load_partial_state_dict(pretrain_model, ckpt_path, logger, map_location='cpu', verbose=True)
                else:
                    logger.warning(f"Checkpoint {ckpt_path} 不存在，使用随机初始化。")

                pretrain_model = pretrain_model.cuda()
                attach_input_projection_to_pretrain(pretrain_model, feature_size, logger)

                accuracy_list = []
                ring_loss_history = []

                for i in range(args.test_times):
                    logger.info("-" * 40)
                    logger.info(f"[Task {i + 1}/{args.test_times}] Mode: {mode_name}")

                    seed_everything(seed_base + i)
                    dataset = dataset.shuffle()

                    n_total = len(dataset)
                    n_train = int(0.5 * n_total)
                    n_val = int(0.8 * n_total)

                    full_train_dataset = dataset[:n_train]
                    val_dataset = dataset[n_train:n_val]
                    test_dataset = dataset[n_val:]

                    if use_full_supervision:
                        train_dataset = full_train_dataset
                        suffix = "full"
                        logger.info(f"使用全量 Train Set (Memory): {len(train_dataset)} graphs")
                    else:
                        try:
                            few_list = sample_fewshot_by_class(full_train_dataset, num_classes, k=shot_k,
                                                               seed=seed_base + i)
                            train_dataset = ListDataset(few_list)
                            suffix = f"shot{shot_k}"
                            logger.info(f"使用 Few-shot Train Set (Memory): {len(train_dataset)} graphs")
                        except ValueError as e:
                            logger.error(f"采样失败: {e}，跳过当前 Task")
                            continue

                    resource_dataset = train_dataset if args.resource_on == 'train' else val_dataset
                    logger.info(f"Resource split: {args.resource_on}, graphs={len(resource_dataset)}")

                    if args.model_variant == 'legacy':
                        rag_model = LegacyRAGraph(
                            pretrain_model,
                            resource_dataset=train_dataset,
                            feture_size=feature_size,
                            num_class=num_classes,
                            emb_size=256,
                            finetune=True,
                            noise_finetune=False,
                            dataset_name=dataset_name
                        ).cuda()
                    elif args.model_variant == 'retag':
                        rag_model = ReTAGRAGraph(
                            pretrain_model,
                            resource_dataset=resource_dataset,
                            feture_size=feature_size,
                            num_class=num_classes,
                            emb_size=256,
                            finetune=True,
                            noise_finetune=False,
                            dataset_name=dataset_name,
                            ring_weight=args.ring_weight,
                            query_graph_hop=args.query_hop,
                            retrieve_num=args.retrieve_num,
                            fusion_gamma=args.fusion_gamma,
                            max_ring=args.max_ring
                        ).cuda()
                    elif args.model_variant == 'memory':
                        rag_model = MemoryRAGraph(
                            pretrain_model,
                            resource_dataset=resource_dataset,
                            feture_size=feature_size,
                            num_class=num_classes,
                            emb_size=256,
                            finetune=True,
                            noise_finetune=False,
                            dataset_name=dataset_name,
                            ring_weight=args.ring_weight,
                            query_graph_hop=args.query_hop,
                            retrieve_num=args.retrieve_num,
                            fusion_gamma=args.fusion_gamma,
                            max_ring=args.max_ring,
                            use_diff_lifting=args.use_diff_lifting,
                            use_task_rerank=args.use_task_rerank,
                            use_memory_reflection=args.use_memory_reflection,
                        ).cuda()

                    optimizer = torch.optim.Adam(rag_model.parameters(), lr=args.lr)
                    best_loss = float('inf')
                    os.makedirs("modelset", exist_ok=True)
                    finetune_model_name = f"modelset/finetune_rag_model_{dataset_name}_{suffix}_{i}.pkl"

                    # 核心机制：用 Validation 做 Query 微调权重
                    finetune_dataset = train_dataset if args.finetune_on == 'train' else val_dataset
                    finetune_loader = DataLoader(finetune_dataset, batch_size=args.batch_size, shuffle=True)
                    logger.info(f"Finetune split: {args.finetune_on}, graphs={len(finetune_dataset)}")

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats(device)

                    epoch_times = []
                    t_train_start = time.perf_counter()
                    task_ring_loss_log = []

                    for epoch in range(args.epochs):
                        t_epoch_start = time.perf_counter()
                        rag_model.train()
                        total_loss, total_cls, total_ring, valid_batches = 0.0, 0.0, 0.0, 0
                        total_valid_ring_graphs, total_rings = 0, 0

                        # 使用 tqdm，并将输出重定向到 stdout 以便 tmux 查看
                        for data in tqdm(finetune_loader, desc=f"T{i + 1} E{epoch}", leave=False):
                            features, adj, labels, complex_batch, batch = process_tu_dataset(
                                data, num_classes, feature_size, max_ring=args.max_ring
                            )
                            optimizer.zero_grad()

                            ret = rag_model.forward_with_loss(features, adj, complex_batch=complex_batch, label=labels,
                                                              batch=batch)
                            if ret is None: continue

                            loss, logits, debug_info = ret

                            ring_loss = debug_info.get("ring_loss", torch.tensor(0.0, device=device))
                            if not isinstance(ring_loss, torch.Tensor):
                                ring_loss = torch.tensor(ring_loss, device=device)
                            if args.model_variant == 'legacy':
                                loss = loss + args.ring_weight * ring_loss

                            if torch.isnan(loss):
                                logger.warning("Loss is NaN，跳过梯度更新")
                                continue

                            loss.backward()
                            optimizer.step()

                            total_loss += loss.item()
                            cls_loss = debug_info.get("cls_loss", loss)
                            total_cls += cls_loss.item() if isinstance(cls_loss, torch.Tensor) else float(cls_loss)
                            total_ring += ring_loss.item()
                            total_valid_ring_graphs += int(debug_info.get("valid_ring_graphs", 0))
                            total_rings += int(debug_info.get("total_rings", 0))
                            valid_batches += 1

                        avg_loss = total_loss / valid_batches if valid_batches > 0 else float('inf')
                        avg_cls_loss = total_cls / valid_batches if valid_batches > 0 else float('inf')
                        avg_ring_loss = total_ring / valid_batches if valid_batches > 0 else 0.0
                        task_ring_loss_log.append(avg_ring_loss)

                        epoch_time = time.perf_counter() - t_epoch_start
                        epoch_times.append(epoch_time)

                        logger.info(
                            f"[Epoch {epoch}] time={epoch_time:.2f}s, loss={avg_loss:.4f}, "
                            f"cls_loss={avg_cls_loss:.4f}, ring_loss={avg_ring_loss:.4f}, "
                            f"ring_graphs={total_valid_ring_graphs}, rings={total_rings}")

                        current_metric = avg_loss if args.save_metric == 'loss' else avg_cls_loss
                        if current_metric < best_loss:
                            best_loss = current_metric
                            torch.save(rag_model.state_dict(), finetune_model_name)

                    t_train_end = time.perf_counter()
                    total_train_time = t_train_end - t_train_start
                    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
                    peak_mem_gb = torch.cuda.max_memory_allocated(
                        device) / 1024 ** 3 if torch.cuda.is_available() else 0.0

                    logger.info(
                        f"[Task {i + 1}] 训练完成. Avg epoch: {avg_epoch_time:.2f}s, Total: {total_train_time / 60:.2f}min, Peak Mem: {peak_mem_gb:.2f}GB")

                    # 测试阶段
                    if os.path.exists(finetune_model_name):
                        rag_model.load_state_dict(torch.load(finetune_model_name))

                    rag_model.eval()
                    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
                    correct, total = 0, 0

                    with torch.no_grad():
                        for data in test_loader:
                            features, adj, labels, complex_batch, batch = process_tu_dataset(
                                data, num_classes, feature_size, max_ring=args.max_ring
                            )
                            logits = rag_model(features, adj, complex_batch=complex_batch, batch=batch)
                            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                            correct += torch.sum(preds == labels).item()
                            total += labels.size(0)

                    acc = 100.0 * correct / total if total > 0 else 0
                    accuracy_list.append(acc)
                    logger.info(f"[Task {i + 1}] Test Accuracy: {acc:.4f}%")
                    ring_loss_history.append(task_ring_loss_log)

                # 最终结果汇总
                logger.info("=" * 60)
                logger.info(f"Dataset: {dataset_name} | Mode: {mode_name}")
                accs = np.array(accuracy_list)
                logger.info(f"Mean: {accs.mean():.4f}  Std: {accs.std():.4f}")

                os.makedirs("results", exist_ok=True)
                with open(f"results/finetune_rag_{dataset_name}_{suffix}.json", "w") as f:
                    json.dump({"mean": float(accs.mean()), "std": float(accs.std()), "accuracy": accuracy_list}, f,
                              indent=4)

                # 绘图保存
                plt.figure()
                for idx, task_ring in enumerate(ring_loss_history):
                    plt.plot(task_ring, label=f'Task {idx + 1}')
                plt.xlabel("Epoch")
                plt.ylabel("Ring Loss")
                plt.title(f"Ring Loss ({mode_name}) - {dataset_name}")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"results/ring_loss_curve_{dataset_name}_{suffix}.png")
                plt.close()

    logger.info("========== 实验全部结束 ==========")


if __name__ == "__main__":
    main()
