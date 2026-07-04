import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from preprompt import PrePrompt
from RAGraph import RAGraph
from ragraph_utils import seed_everything, process_tu_dataset
from collections import defaultdict
import random

# 参数设置
dataset_name = "ENZYMES"
batch_size = 16
test_times = 5
downstream_epochs = 200
lr = 0.001
ring_loss_weight = 0
k_shot = 5  # 每类 few-shot 样本数

# Few-shot 样本选择函数
# Few-shot 样本选择函数
def sample_few_shot(dataset, num_classes, k_shot, seed=42):
    random.seed(seed)
    class_to_graphs = defaultdict(list)
    for graph in dataset:
        label = int(graph.y.item())
        class_to_graphs[label].append(graph)

    few_shot_graphs = []
    for label in range(num_classes):
        candidates = class_to_graphs[label]
        if len(candidates) >= k_shot:
            few_shot_graphs.extend(random.sample(candidates, k_shot))
        else:
            few_shot_graphs.extend(candidates)
            print(f"[WARN] Class {label} only has {len(candidates)} samples (expected {k_shot})")

    return few_shot_graphs

# 封装 few-shot 为可访问属性的 dataset 对象
from torch_geometric.data import InMemoryDataset

class FewShotDataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    @property
    def num_node_attributes(self):
        return self[0].x.size(1)  # 修复这里，返回特征维度

    @property
    def num_classes(self):
        return len(set([int(data.y.item()) for data in self.data_list]))


# 数据准备
dataset = TUDataset(root='data', name=dataset_name, use_node_attr=True)
assert dataset[0].x is not None, "Your dataset must have node features (x)"
feature_size = dataset[0].x.shape[1]
num_classes = dataset.num_classes
print(f"[INFO] feature_size: {feature_size}, num_classes: {num_classes}")

# 加载预训练模型
pretrain_model = PrePrompt(feature_size, 256, 'prelu', 1, 0.3, use_proj=True)
pretrain_model.load_state_dict(torch.load(f'modelset/model_{dataset_name}.pkl'))
pretrain_model = pretrain_model.cuda()

# 微调与评估
accuracy_list = []
ring_loss_history = []

for i in range(test_times):
    seed_everything(3407 + i)
    dataset = dataset.shuffle()

    full_train_dataset = dataset[:int(0.5 * len(dataset))]
    val_dataset = dataset[int(0.5 * len(dataset)):int(0.8 * len(dataset))]
    test_dataset = dataset[int(0.8 * len(dataset)):]

    train_dataset = FewShotDataset(sample_few_shot(full_train_dataset, num_classes, k_shot, seed=3407 + i))

    rag_model = RAGraph(

        pretrain_model,
        resource_dataset=train_dataset,
        feture_size=feature_size,
        num_class=num_classes,
        emb_size=256,
        finetune=True,
        noise_finetune=False,
    ).cuda()
    rag_model.toy_graph_base.show()
    optimizer = torch.optim.Adam(rag_model.parameters(), lr=lr)
    best_loss = float('inf')
    finetune_model_name = f"modelset/finetune_rag_model_{dataset_name}_fewshot{k_shot}_{i}.pkl"
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    task_ring_loss_log = []

    for epoch in range(downstream_epochs):
        rag_model.train()
        total_loss = 0.0
        total_ring = 0.0

        for data in tqdm(val_loader, desc=f'Task {i + 1} Epoch {epoch}'):
            features, adj, labels, complex_batch, batch = process_tu_dataset(data, num_classes, feature_size)
            optimizer.zero_grad()

            loss, logits, debug_info = rag_model.forward_with_loss(
                features, adj, complex_batch=complex_batch, label=labels, batch=batch
            )
            ring_loss = debug_info.get("ring_loss", torch.tensor(0.0, device=logits.device))
            if not isinstance(ring_loss, torch.Tensor):
                ring_loss = torch.tensor(ring_loss, device=logits.device)

            ring_weight = 3.5
            loss = loss + ring_weight * ring_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ring += ring_loss.item()

        avg_loss = total_loss / len(val_loader)
        avg_ring_loss = total_ring / len(val_loader)
        task_ring_loss_log.append(avg_ring_loss)

        print(f"[EPOCH {epoch}] loss = {avg_loss:.4f}, ring_loss = {avg_ring_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(rag_model.state_dict(), finetune_model_name)

    ring_loss_history.append(task_ring_loss_log)

    # 测试
    rag_model.load_state_dict(torch.load(finetune_model_name))
    rag_model.eval()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    correct, total = 0, 0
    for data in test_loader:
        features, adj, labels, complex_batch, batch = process_tu_dataset(data, num_classes, feature_size)
        logits = rag_model(features, adj, complex_batch=complex_batch, batch=batch)
        preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

    acc = 100 * correct / total
    accuracy_list.append(acc)
    print(f"[Task {i + 1}] Accuracy: {acc:.4f}%")

# 结果输出
print("-" * 100)
print(" 所有任务准确率:")
for i, acc in enumerate(accuracy_list):
    print(f"Task {i + 1}: {acc:.4f}%")

accs = np.array(accuracy_list)
print(f"\nmean: {accs.mean():.4f}%")
print(f"std: {accs.std():.4f}%")
print("-" * 100)

# 保存结果
os.makedirs("results", exist_ok=True)
with open(f"results/finetune_rag_{dataset_name}_fewshot{k_shot}.json", "w") as f:
    json.dump({
        "mean": float(accs.mean()),
        "std": float(accs.std()),
        "accuracy": accuracy_list,
        "few_shot": k_shot
    }, f, indent=4)

# 绘图
plt.figure()
for i, task_ring_loss in enumerate(ring_loss_history):
    plt.plot(task_ring_loss, label=f'Task {i + 1}')
plt.xlabel("Epoch")
plt.ylabel("Ring Loss")
plt.title("Ring Contrastive Loss over Epochs (All Tasks)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"results/ring_loss_curve_all_{dataset_name}_fewshot{k_shot}.png")
plt.close()

for i, task_ring_loss in enumerate(ring_loss_history):
    plt.figure()
    plt.plot(task_ring_loss, color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Ring Loss")
    plt.title(f"Ring Contrastive Loss - Task {i + 1}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/ring_loss_curve_task{i+1}_{dataset_name}_fewshot{k_shot}.png")
    plt.close()
