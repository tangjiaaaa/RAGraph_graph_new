import argparse
import datetime
import json
import os

import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm
from torch_geometric.datasets import TUDataset

try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader

from preprompt import PrePrompt
from RAGraph2 import RAGraph
from ragraph_utils import seed_everything
from utils import process


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def label_distribution(dataset, feature_size, max_ring, num_classes):
    labels_all = []
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    for data in loader:
        _, _, labels, _ = process_batch(data, feature_size, max_ring)
        labels_all.append(labels.cpu())
    labels = torch.cat(labels_all, dim=0)
    return torch.bincount(labels, minlength=num_classes).tolist()


def process_batch(data, feature_size, max_ring):
    features, adj, node_labels, complex_obj = process.process_tu(data, feature_size, max_ring=max_ring)
    adj = process.normalize_adj(adj + sp.eye(adj.shape[0])).todense()

    features = torch.FloatTensor(features).cuda()
    adj = torch.FloatTensor(np.asarray(adj)).cuda()
    node_labels = torch.argmax(torch.FloatTensor(node_labels), dim=1).long().cuda()
    complex_obj = complex_obj.to(features.device)

    return features, adj, node_labels, complex_obj


def class_weights_from_distribution(label_dist, power=1.0, max_weight=None):
    counts = torch.tensor(label_dist, dtype=torch.float32).cuda()
    weights = counts.sum() / counts.clamp_min(1.0)
    if power != 1.0:
        weights = weights.pow(power)
    weights = weights / weights.mean().clamp_min(1e-12)
    if max_weight is not None and max_weight > 0:
        weights = torch.clamp(weights, max=max_weight)
        weights = weights / weights.mean().clamp_min(1e-12)
    return weights


def classification_loss(logits, labels, class_weights=None, loss_type="nll"):
    logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
    if loss_type == "baseline_ce":
        return torch.nn.functional.cross_entropy(logits, labels, weight=class_weights)
    log_probs = torch.log(logits.clamp_min(1e-8))
    return torch.nn.functional.nll_loss(log_probs, labels, weight=class_weights)


def apply_prior_correction(logits, label_dist, alpha):
    if alpha <= 0:
        return logits
    counts = torch.tensor(label_dist, dtype=torch.float32, device=logits.device)
    prior = counts / counts.sum().clamp_min(1.0)
    corrected = logits.clamp_min(1e-8) / prior.clamp_min(1e-8).pow(alpha)
    return corrected / corrected.sum(dim=1, keepdim=True).clamp_min(1e-8)


def balanced_cross_entropy(logits, labels, num_classes, power=1.0, max_weight=None, loss_type="nll"):
    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = class_weights_from_distribution(counts.cpu().tolist(), power=power, max_weight=max_weight)
    return classification_loss(logits, labels, weights, loss_type=loss_type)


def main():
    parser = argparse.ArgumentParser("RAGraph full-supervised finetune with cell complex")
    parser.add_argument("--dataset", type=str, default="ENZYMES")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--ring_weight", type=float, default=0.1)
    parser.add_argument("--max_ring", type=int, default=6)
    parser.add_argument("--test_times", type=int, default=5)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed_base", type=int, default=3407)
    parser.add_argument("--save_metric", type=str, default="loss", choices=["loss", "train_acc", "balanced_acc"])
    parser.add_argument("--loss_type", type=str, default="nll", choices=["nll", "baseline_ce"])
    parser.add_argument("--balanced_ce", action="store_true")
    parser.add_argument("--class_weight_mode", type=str, default="none", choices=["none", "batch", "global"])
    parser.add_argument("--class_weight_power", type=float, default=1.0)
    parser.add_argument("--class_weight_max", type=float, default=None)
    parser.add_argument("--label_weight", type=float, default=None)
    parser.add_argument("--retrieve_weight", type=float, default=None)
    parser.add_argument("--prior_correction_alpha", type=float, default=0.0)
    parser.add_argument("--prior_correction_train", action="store_true")
    args = parser.parse_args()

    args.save_name = f"modelset/model_{args.dataset}.pkl"
    seed_everything(args.seed)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = os.path.join(args.log_dir, f"finetune_rag524_{args.dataset}_{timestamp}")
    os.makedirs(run_log_dir, exist_ok=True)
    epoch_log_path = os.path.join(run_log_dir, "epochs.jsonl")
    task_log_path = os.path.join(run_log_dir, "tasks.jsonl")

    hid_units = 256
    nonlinearity = "prelu"

    base_dataset = TUDataset(root="data", name=args.dataset, use_node_attr=True)
    feature_size = base_dataset.num_node_attributes
    sample_feature_dim = base_dataset[0].x.size(1)
    num_classes = sample_feature_dim - feature_size
    if num_classes <= 0:
        num_classes = getattr(dataset, "num_node_labels", 0)
    if num_classes <= 0:
        raise ValueError(
            f"Cannot infer node label dimension: sample x dim={sample_feature_dim}, "
            f"num_node_attributes={feature_size}, dataset.num_features={base_dataset.num_features}. "
            "This node-classification pipeline expects node labels to be concatenated after node attributes."
        )
    graph_classes = base_dataset.num_classes
    print(
        f">> dataset={args.dataset} feature_size={feature_size} "
        f"sample_x_dim={sample_feature_dim} node_classes={num_classes} "
        f"graph_classes={graph_classes}"
    )
    with open(os.path.join(run_log_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "save_name": args.save_name,
                "feature_size": feature_size,
                "sample_feature_dim": sample_feature_dim,
                "node_classes": num_classes,
                "graph_classes": graph_classes,
                "dataset_size": len(base_dataset),
                "hid_units": hid_units,
                "nonlinearity": nonlinearity,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    pretrain_model = PrePrompt(feature_size, hid_units, nonlinearity, 1, 0.3)
    pretrain_model.load_state_dict(torch.load(args.save_name))
    pretrain_model = pretrain_model.cuda()

    accuracy_list = []

    for run_id in range(args.test_times):
        task_seed = args.seed_base
        seed_everything(task_seed)
        dataset = base_dataset.shuffle()
        train_dataset = dataset[:int(0.5 * len(dataset))]
        val_dataset = dataset[int(0.5 * len(dataset)):int(0.8 * len(dataset))]
        test_dataset = dataset[int(0.8 * len(dataset)):]
        append_jsonl(
            task_log_path,
            {
                "event": "split",
                "task": run_id,
                "seed": task_seed,
                "train_size": len(train_dataset),
                "val_size": len(val_dataset),
                "test_size": len(test_dataset),
                "train_label_dist": label_distribution(train_dataset, feature_size, args.max_ring, num_classes),
                "finetune_label_dist": label_distribution(val_dataset, feature_size, args.max_ring, num_classes),
                "test_label_dist": label_distribution(test_dataset, feature_size, args.max_ring, num_classes),
            },
        )

        rag_model = RAGraph(
            pretrain_model,
            resource_dataset=train_dataset,
            feture_size=feature_size,
            num_class=num_classes,
            emb_size=hid_units,
            finetune=True,
            noise_finetune=False,
        ).cuda()
        if args.label_weight is not None:
            rag_model.label_weight = args.label_weight
        if args.retrieve_weight is not None:
            rag_model.retrieve_weight = args.retrieve_weight

        print(f"\n=================== [Task {run_id + 1} / {args.test_times}] ===================")

        rag_model.train()
        best_loss = float("inf")
        finetune_model_name = f"modelset/finetune_rag_model_{args.dataset}_{run_id}.pkl"
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
        opt = torch.optim.Adam(rag_model.parameters(), lr=args.lr)
        finetune_label_dist = label_distribution(val_dataset, feature_size, args.max_ring, num_classes)
        global_class_weights = class_weights_from_distribution(
            finetune_label_dist,
            power=args.class_weight_power,
            max_weight=args.class_weight_max,
        )
        use_class_weight_mode = args.class_weight_mode
        if args.balanced_ce and use_class_weight_mode == "none":
            use_class_weight_mode = "batch"
        print(
            f"[Task {run_id + 1}] class_weight_mode={use_class_weight_mode} "
            f"finetune_label_dist={finetune_label_dist} "
            f"global_class_weights={global_class_weights.detach().cpu().tolist()} "
            f"class_weight_power={args.class_weight_power} class_weight_max={args.class_weight_max} "
            f"loss_type={args.loss_type} "
            f"retrieve_weight={rag_model.retrieve_weight} label_weight={rag_model.label_weight} "
            f"prior_correction_alpha={args.prior_correction_alpha} "
            f"prior_correction_train={args.prior_correction_train}"
        )

        for epoch in range(args.epochs):
            total_loss = 0.0
            valid_batch = 0
            train_correct = 0
            train_total = 0
            train_correct_by_class = torch.zeros(num_classes, dtype=torch.float32, device="cuda")
            train_label_counts = torch.zeros(num_classes, dtype=torch.float32, device="cuda")

            for data in tqdm(val_loader, desc=f"Epoch {epoch}", ncols=80, leave=False):
                features, adj, node_labels, complex_obj = process_batch(data, feature_size, args.max_ring)

                opt.zero_grad()
                logits, ring_loss = rag_model(
                    features,
                    adj,
                    complex_obj=complex_obj,
                    return_ring_loss=True,
                )
                logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
                if args.prior_correction_train:
                    logits = apply_prior_correction(logits, finetune_label_dist, args.prior_correction_alpha)
                ring_loss = torch.nan_to_num(ring_loss, nan=0.0, posinf=0.0, neginf=0.0)

                if use_class_weight_mode == "global":
                    cls_loss = classification_loss(
                        logits,
                        node_labels,
                        global_class_weights,
                        loss_type=args.loss_type,
                    )
                elif use_class_weight_mode == "batch":
                    cls_loss = balanced_cross_entropy(
                        logits,
                        node_labels,
                        num_classes,
                        power=args.class_weight_power,
                        max_weight=args.class_weight_max,
                        loss_type=args.loss_type,
                    )
                else:
                    cls_loss = classification_loss(logits, node_labels, loss_type=args.loss_type)
                loss = cls_loss + args.ring_weight * ring_loss
                if not torch.isfinite(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(rag_model.parameters(), max_norm=1.0)
                opt.step()

                total_loss += loss.item()
                valid_batch += 1
                pred = torch.argmax(logits.detach(), dim=1)
                train_correct += torch.sum(pred == node_labels).item()
                train_total += node_labels.size(0)
                train_label_counts += torch.bincount(node_labels, minlength=num_classes).float()
                train_correct_by_class += torch.bincount(
                    node_labels[pred == node_labels],
                    minlength=num_classes,
                ).float()

            epoch_loss = total_loss / valid_batch if valid_batch > 0 else float("inf")
            epoch_train_acc = 100.0 * train_correct / train_total if train_total > 0 else 0.0
            epoch_balanced_acc = (
                100.0
                * (
                    train_correct_by_class
                    / train_label_counts.clamp_min(1.0)
                ).mean().item()
            )
            print(
                f"[Epoch {epoch:03d}/{args.epochs}] loss={epoch_loss:.6f} "
                f"train_acc={epoch_train_acc:.4f}% "
                f"balanced_acc={epoch_balanced_acc:.4f}% valid_batches={valid_batch}"
            )
            append_jsonl(
                epoch_log_path,
                {
                    "task": run_id,
                    "epoch": epoch,
                    "loss": float(epoch_loss),
                    "train_acc": float(epoch_train_acc),
                    "balanced_acc": float(epoch_balanced_acc),
                    "valid_batches": valid_batch,
                    "best_loss_before_update": float(best_loss),
                },
            )

            if args.save_metric == "loss":
                current_metric = epoch_loss
            elif args.save_metric == "balanced_acc":
                current_metric = -epoch_balanced_acc
            else:
                current_metric = -epoch_train_acc
            if current_metric < best_loss:
                best_loss = current_metric
                torch.save(rag_model.state_dict(), finetune_model_name)

        rag_model.load_state_dict(torch.load(finetune_model_name))
        rag_model.eval()
        rag_model.toy_graph_base.build_toy_graph(val_dataset)

        correct = 0
        total = 0
        pred_counts = torch.zeros(num_classes, dtype=torch.long)
        label_counts = torch.zeros(num_classes, dtype=torch.long)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        with torch.no_grad():
            for data in test_loader:
                features, adj, node_labels, complex_obj = process_batch(data, feature_size, args.max_ring)
                logits = rag_model(features, adj, complex_obj=complex_obj)
                logits = apply_prior_correction(logits, finetune_label_dist, args.prior_correction_alpha)
                pred = torch.argmax(logits, dim=1)
                correct += torch.sum(pred == node_labels).item()
                total += node_labels.size(0)
                pred_counts += torch.bincount(pred.cpu(), minlength=num_classes)
                label_counts += torch.bincount(node_labels.cpu(), minlength=num_classes)

        accuracy = 100.0 * correct / total if total > 0 else 0.0
        print(
            f"[*] Task {run_id + 1} done | best_metric={best_loss:.4f} | acc={accuracy:.4f}% "
            f"| labels={label_counts.tolist()} preds={pred_counts.tolist()}"
        )
        accuracy_list.append(accuracy)
        append_jsonl(
            task_log_path,
            {
                "event": "result",
                "task": run_id,
                "best_loss": float(best_loss),
                "accuracy": float(accuracy),
                "correct": int(correct),
                "total": int(total),
                "test_label_dist": label_counts.tolist(),
                "test_pred_dist": pred_counts.tolist(),
                "checkpoint": finetune_model_name,
            },
        )

    accs = np.array(accuracy_list)
    mean_acc = accs.mean()
    std_acc = accs.std()
    print("=" * 50)
    print(f"Dataset: {args.dataset}")
    print(f"Mean: {mean_acc:.4f}%")
    print(f"Std : {std_acc:.4f}%")
    print(f"All : {accuracy_list}")
    print("=" * 50)

    os.makedirs("results", exist_ok=True)
    summary = {"mean": float(mean_acc), "std": float(std_acc), "accuracy": accuracy_list}
    with open(f"results/finetune_rag_{args.dataset}.json", "w") as f:
        json.dump(summary, f, indent=4)
    with open(os.path.join(run_log_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[LOG] saved run logs to {run_log_dir}")


if __name__ == "__main__":
    main()
