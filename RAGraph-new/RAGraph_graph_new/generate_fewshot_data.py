import os
import torch
import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_adj

# 数据集名称
datasets = ["COX2", "PROTEINS", "BZR", "ENZYMES"]

# 设置每个 shot 包含的样本数
shots = [1, 2, 3, 4, 5]

# 生成文件的路径
base_dir = "data/fewshot"


# 创建并保存每个 shot 的数据
def generate_shot_files(dataset_name, shot_k):
    # 加载数据集
    dataset = TUDataset(root=f"{base_dir}_{dataset_name}_graph", name=dataset_name, use_node_attr=True)

    # 随机选择 shot_k 个样本作为训练集
    num_classes = dataset.num_classes
    dataset = dataset.shuffle()

    # 选择每个类中的前 shot_k 个图样本
    label_to_graphs = {label: [] for label in range(num_classes)}
    for data in dataset:
        label = int(data.y.item())
        label_to_graphs[label].append(data)

    selected_graphs = []
    for label, graphs in label_to_graphs.items():
        selected_graphs.extend(graphs[:shot_k])

    # 组织成 Batch 数据
    from torch_geometric.data import Batch
    batch = Batch.from_data_list(selected_graphs)

    # 转换为 dense_adj，获取每个图的邻接矩阵和特征
    adj = to_dense_adj(batch.edge_index, batch=batch.batch)
    feature = batch.x
    labels = batch.y
    graph_len = batch.batch

    # 保存文件
    shot_dir = f"{base_dir}_{dataset_name}_graph/{shot_k}shot_{dataset_name}_graph"

    # 创建目录，如果不存在则创建
    os.makedirs(f"{shot_dir}/testset", exist_ok=True)

    # 保存数据文件
    torch.save(adj, f"{shot_dir}/testset/adj.pt")
    torch.save(feature, f"{shot_dir}/testset/feature.pt")
    torch.save(labels, f"{shot_dir}/testset/labels.pt")
    torch.save(graph_len, f"{shot_dir}/testset/graph_len.pt")
    print(f"Generated {shot_k}-shot data for {dataset_name}")


# 主函数：为所有数据集生成 1 到 5 shot 的数据文件
def main():
    for dataset_name in datasets:
        for shot_k in shots:
            print(f"Generating {shot_k}-shot data for {dataset_name}...")
            generate_shot_files(dataset_name, shot_k)
            print(f"{shot_k}-shot data generation complete for {dataset_name}")


if __name__ == "__main__":
    main()
