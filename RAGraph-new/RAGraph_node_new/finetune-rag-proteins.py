import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.datasets import TUDataset

try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader

from preprompt import PrePrompt
from RAGraph import RAGraph
from pretrain3 import ring_contrastive_loss
from ragraph_utils import process_tu_dataset, seed_everything
from utils import process


def graph_node_label_counts(graph, feature_size, num_classes):
    labels = torch.argmax(graph.x[:, feature_size:], dim=1)
    return torch.bincount(labels.cpu(), minlength=num_classes).long()


def split_dataset(dataset, feature_size, num_classes, seed, mode):
    if mode == "random":
        seed_everything(seed)
        shuffled = dataset.shuffle()
        train_dataset = shuffled[: int(0.5 * len(shuffled))]
        val_dataset = shuffled[int(0.5 * len(shuffled)) : int(0.8 * len(shuffled))]
        test_dataset = shuffled[int(0.8 * len(shuffled)) :]
        return train_dataset, val_dataset, test_dataset

    rng = np.random.RandomState(seed)
    groups = {label: [] for label in range(num_classes)}
    for idx in range(len(dataset)):
        counts = graph_node_label_counts(dataset[idx], feature_size, num_classes)
        dominant = int(torch.argmax(counts).item())
        groups[dominant].append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for indices in groups.values():
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(0.5 * n)
        n_val = int(0.3 * n)
        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train : n_train + n_val])
        test_idx.extend(indices[n_train + n_val :])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return dataset[train_idx], dataset[val_idx], dataset[test_idx]


def label_counts_from_dataset(dataset, feature_size, num_classes, batch_size):
    counts = torch.zeros(num_classes, dtype=torch.long)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for data in loader:
        _, _, node_labels = process_tu_dataset(data, feature_size)
        labels = torch.argmax(node_labels, dim=1).detach().cpu()
        counts += torch.bincount(labels, minlength=num_classes)
    return counts.tolist()


def batch_complex(data, feature_size, max_ring):
    _, _, _, complex_obj = process.process_tu(data, feature_size, max_ring=max_ring)
    return complex_obj


def compute_ring_loss(rag_model, features, adj, complex_obj, max_ring_samples):
    if (
        complex_obj is None
        or not hasattr(complex_obj, "cochains")
        or 2 not in complex_obj.cochains
        or complex_obj.cochains[2] is None
        or complex_obj.cochains[2].boundary_index is None
        or complex_obj.cochains[2].boundary_index.numel() == 0
    ):
        return torch.tensor(0.0, requires_grad=True, device=features.device)

    node_emb = rag_model.pretrain_model.inference(features, adj)
    ring_boundary = complex_obj.cochains[2].boundary_index.to(features.device)
    edge_index = complex_obj.cochains[1].boundary_index.to(features.device)
    loss = ring_contrastive_loss(node_emb, ring_boundary, edge_index, max_rings=max_ring_samples)
    return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)


def compute_hidden_ring_loss(hidden, complex_obj, max_ring_samples):
    if (
        complex_obj is None
        or not hasattr(complex_obj, "cochains")
        or 2 not in complex_obj.cochains
        or complex_obj.cochains[2] is None
        or complex_obj.cochains[2].boundary_index is None
        or complex_obj.cochains[2].boundary_index.numel() == 0
    ):
        return torch.tensor(0.0, requires_grad=True, device=hidden.device)

    ring_boundary = complex_obj.cochains[2].boundary_index.to(hidden.device)
    edge_index = complex_obj.cochains[1].boundary_index.to(hidden.device)
    loss = ring_contrastive_loss(hidden, ring_boundary, edge_index, max_rings=max_ring_samples)
    return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)


def main():
    parser = argparse.ArgumentParser("Baseline finetune plus optional cell ring loss")
    parser.add_argument("--dataset", type=str, default="PROTEINS")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--test_times", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed_base", type=int, default=0)
    parser.add_argument("--retrieve_num", type=int, default=4)
    parser.add_argument("--split_mode", type=str, default="random", choices=["random", "stratified"])
    parser.add_argument("--ring_weight", type=float, default=0.0)
    parser.add_argument("--max_ring", type=int, default=6)
    parser.add_argument("--max_ring_samples", type=int, default=100)
    parser.add_argument("--ring_target", type=str, default="hidden", choices=["hidden", "pretrain"])
    parser.add_argument("--save_metric", type=str, default="balanced_loss", choices=["loss", "balanced_loss"])
    parser.add_argument("--balance_lambda", type=float, default=0.15)
    parser.add_argument("--log_dir", type=str, default="logs")
    args = parser.parse_args()
    args.save_name = f"modelset/model_{args.dataset}.pkl"

    seed_everything(args.seed)
    hid_units = 256
    nonlinearity = "prelu"

    dataset = TUDataset(root="data", name=args.dataset, use_node_attr=True)
    feature_size = dataset.num_node_attributes
    sample_feature_dim = dataset[0].x.size(1)
    num_classes = sample_feature_dim - feature_size
    if num_classes <= 0:
        num_classes = 3

    print("-" * 100)
    print(args)
    print(
        f"dataset={args.dataset} feature_size={feature_size} "
        f"sample_x_dim={sample_feature_dim} node_classes={num_classes}"
    )
    print("-" * 100)

    pretrain_model = PrePrompt(feature_size, hid_units, nonlinearity, 1, 0.3)
    pretrain_model.load_state_dict(torch.load(args.save_name))
    pretrain_model = pretrain_model.cuda()

    accuracy_list = []
    task_records = []
    for i in range(args.test_times):
        task_seed = args.seed_base + i
        seed_everything(task_seed)
        print("-" * 100)

        train_dataset, val_dataset, test_dataset = split_dataset(
            dataset,
            feature_size,
            num_classes,
            task_seed,
            args.split_mode,
        )

        train_label_counts = label_counts_from_dataset(train_dataset, feature_size, num_classes, args.batch_size)
        val_label_counts = label_counts_from_dataset(val_dataset, feature_size, num_classes, args.batch_size)
        test_label_counts = label_counts_from_dataset(test_dataset, feature_size, num_classes, args.batch_size)

        print(
            f"tasknum={i + 1} seed={task_seed} split_mode={args.split_mode} "
            f"train_labels={train_label_counts} "
            f"val_labels={val_label_counts} "
            f"test_labels={test_label_counts}"
        )

        rag_model = RAGraph(
            pretrain_model,
            resource_dataset=train_dataset,
            feture_size=feature_size,
            num_class=num_classes,
            emb_size=hid_units,
            finetune=True,
            noise_finetune=False,
            retrieve_num=args.retrieve_num,
        ).cuda()

        rag_model.train()
        best_metric = float("inf")
        best_loss = float("inf")
        best_balance_penalty = float("inf")
        best_pred01_ratio = 0.0
        finetune_model_name = f"modelset/finetune_baseline_cell_balanced_{args.dataset}_{i}.pkl"
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
        opt = torch.optim.Adam(rag_model.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            total_loss = 0.0
            total_cls_loss = 0.0
            total_ring_loss = 0.0
            valid_batch = 0
            train_correct = 0
            train_total = 0
            epoch_pred_counts = torch.zeros(num_classes, dtype=torch.long)

            for data in tqdm(val_loader, desc=f"epoch {epoch}", ncols=80, leave=False):
                features, adj, node_labels = process_tu_dataset(data, feature_size)
                node_labels = torch.argmax(node_labels, dim=1)

                opt.zero_grad()
                if args.ring_weight > 0 and args.ring_target == "hidden":
                    logits, hidden = rag_model(features, adj, return_hidden=True)
                else:
                    logits = rag_model(features, adj)
                    hidden = None
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    continue

                cls_loss = torch.nn.functional.cross_entropy(logits, node_labels)
                ring_loss = torch.tensor(0.0, requires_grad=True, device=features.device)
                if args.ring_weight > 0:
                    complex_obj = batch_complex(data, feature_size, args.max_ring)
                    if args.ring_target == "hidden":
                        ring_loss = compute_hidden_ring_loss(hidden, complex_obj, args.max_ring_samples)
                    else:
                        ring_loss = compute_ring_loss(
                            rag_model,
                            features,
                            adj,
                            complex_obj,
                            args.max_ring_samples,
                        )

                loss = cls_loss + args.ring_weight * ring_loss
                if not torch.isfinite(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(rag_model.parameters(), max_norm=1.0)
                opt.step()

                total_loss += loss.item()
                total_cls_loss += cls_loss.item()
                total_ring_loss += ring_loss.item()
                valid_batch += 1
                pred = torch.argmax(logits.detach(), dim=1)
                train_correct += torch.sum(pred == node_labels).item()
                train_total += node_labels.size(0)
                epoch_pred_counts += torch.bincount(pred.cpu(), minlength=num_classes)

            epoch_loss = total_loss / valid_batch if valid_batch else float("inf")
            epoch_cls_loss = total_cls_loss / valid_batch if valid_batch else float("inf")
            epoch_ring_loss = total_ring_loss / valid_batch if valid_batch else 0.0
            train_acc = 100.0 * train_correct / train_total if train_total else 0.0
            pred01_total = int(epoch_pred_counts[0] + epoch_pred_counts[1]) if num_classes >= 2 else 0
            pred01_ratio = float(epoch_pred_counts[0]) / pred01_total if pred01_total else 0.5
            balance_penalty = abs(pred01_ratio - 0.5)
            save_metric = epoch_loss
            if args.save_metric == "balanced_loss":
                save_metric = epoch_loss + args.balance_lambda * balance_penalty
            print(
                f"epoch={epoch} loss={epoch_loss:.6f} cls_loss={epoch_cls_loss:.6f} "
                f"ring_loss={epoch_ring_loss:.6f} train_acc={train_acc:.4f}% "
                f"pred01_ratio={pred01_ratio:.4f} balance_penalty={balance_penalty:.4f} "
                f"save_metric={save_metric:.6f}"
            )
            if save_metric < best_metric:
                best_metric = save_metric
                best_loss = epoch_loss
                best_balance_penalty = balance_penalty
                best_pred01_ratio = pred01_ratio
                torch.save(rag_model.state_dict(), finetune_model_name)

        print("-" * 100)
        print("best_loss:", best_loss)
        print(
            f"best_metric: {best_metric:.6f} "
            f"best_pred01_ratio: {best_pred01_ratio:.4f} "
            f"best_balance_penalty: {best_balance_penalty:.4f}"
        )
        rag_model.load_state_dict(torch.load(finetune_model_name))
        rag_model.eval()
        rag_model.toy_graph_base.build_toy_graph(val_dataset)
        rag_model.toy_graph_base.show()

        correct = 0
        total = 0
        pred_counts = torch.zeros(num_classes, dtype=torch.long)
        label_counts = torch.zeros(num_classes, dtype=torch.long)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        with torch.no_grad():
            for data in test_loader:
                features, adj, node_labels = process_tu_dataset(data, feature_size)
                logits = rag_model(features, adj)
                pred = torch.argmax(logits, dim=1)
                labels = torch.argmax(node_labels, dim=1)
                correct += torch.sum(pred == labels).item()
                total += labels.size(0)
                pred_counts += torch.bincount(pred.cpu(), minlength=num_classes)
                label_counts += torch.bincount(labels.cpu(), minlength=num_classes)

        accuracy = 100.0 * correct / total
        print(
            f"accuracy: {accuracy:.4f} "
            f"labels={label_counts.tolist()} preds={pred_counts.tolist()}"
        )
        accuracy_list.append(accuracy)
        task_records.append(
            {
                "tasknum": i + 1,
                "seed": task_seed,
                "best_loss": float(best_loss),
                "best_metric": float(best_metric),
                "best_pred01_ratio": float(best_pred01_ratio),
                "best_balance_penalty": float(best_balance_penalty),
                "accuracy": float(accuracy),
                "train_labels": train_label_counts,
                "val_labels": val_label_counts,
                "test_labels": test_label_counts,
                "labels": label_counts.tolist(),
                "preds": pred_counts.tolist(),
            }
        )

    accs = np.array(accuracy_list)
    best_idx = int(accs.argmax()) if len(accs) else -1
    best_acc = float(accs[best_idx]) if len(accs) else float("nan")
    best_seed = task_records[best_idx]["seed"] if best_idx >= 0 else None
    print("-" * 100)
    print(f"Mean:[{accs.mean():.4f}]")
    print(f"Std :[{accs.std():.4f}]")
    print(f"Best:[{best_acc:.4f}] seed={best_seed} tasknum={best_idx + 1}")
    print(f"All :{accuracy_list}")
    print("-" * 100)

    os.makedirs("results", exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    run_log = {
        "dataset": args.dataset,
        "mean": float(accs.mean()),
        "std": float(accs.std()),
        "best": best_acc,
        "best_seed": best_seed,
        "best_tasknum": best_idx + 1,
        "accuracy": accuracy_list,
        "args": vars(args),
        "feature_size": int(feature_size),
        "sample_feature_dim": int(sample_feature_dim),
        "node_classes": int(num_classes),
        "tasks": task_records,
    }
    with open(f"results/finetune_baseline_cell_{args.dataset}.json", "w") as f:
        json.dump(run_log, f, indent=4)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"finetune_baseline_cell_{args.dataset}_{run_stamp}.json")
    with open(log_path, "w") as f:
        json.dump(run_log, f, indent=4)
    print(f"[LOG] saved run log to {log_path}")


if __name__ == "__main__":
    main()
