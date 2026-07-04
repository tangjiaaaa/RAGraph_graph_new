"""Minimal template for running the memory-enhanced RAGraph.

Put this file under:
    RAGraph-new/RAGraph_graph_new/

Then run it inside the same environment that can run the original RAGraph code.
This template assumes the pretrained checkpoint already exists at:
    modelset/model_{dataset}.pkl
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from preprompt import PrePrompt
from RAGraph_memory import RAGraph
from ragraph_utils import process_tu_dataset, seed_everything


def run(args):
    seed_everything(args.seed)
    dataset = TUDataset(root="data", name=args.dataset, use_node_attr=True).shuffle()
    feature_size = dataset.num_node_attributes
    num_classes = dataset.num_classes

    train_dataset = dataset[: int(0.8 * len(dataset))]
    test_dataset = dataset[int(0.8 * len(dataset)) :]

    pretrain_model = PrePrompt(feature_size, args.hidden, "prelu", 1, 0.3).cuda()
    pretrain_model.load_state_dict(torch.load(f"modelset/model_{args.dataset}.pkl"))
    pretrain_model.eval()

    model = RAGraph(
        pretrain_model,
        resource_dataset=train_dataset,
        feture_size=feature_size,
        num_class=num_classes,
        emb_size=args.hidden,
        finetune=True,
        ring_weight=args.ring_weight,
        retrieval_weight=args.retrieval_weight,
        query_graph_hop=args.hop,
        retrieve_num=args.topk,
        fusion_gamma=args.fusion_gamma,
        max_ring=args.max_ring,
        use_diff_lifting=args.use_diff_lifting,
        use_task_rerank=args.use_task_rerank,
        use_memory_reflection=args.use_memory_reflection,
        memory_utility_weight=args.memory_utility_weight,
    ).cuda()

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for data in train_loader:
            features, adj, labels, complex_batch, batch = process_tu_dataset(
                data, num_classes, feature_size, max_ring=args.max_ring
            )
            optimizer.zero_grad()
            loss, logits, metrics = model.forward_with_loss(
                features, adj, complex_batch, labels, batch=batch
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        if epoch % args.eval_every == 0:
            result = evaluate(model, test_loader, num_classes, feature_size, args.max_ring)
            print(
                f"epoch={epoch:03d} loss={np.mean(losses):.4f} "
                f"test_acc={result['acc']:.4f}"
            )

    result = evaluate(model, test_loader, num_classes, feature_size, args.max_ring)
    os.makedirs("results", exist_ok=True)
    with open(f"results/memory_ragraph_{args.dataset}_seed{args.seed}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(result)


@torch.no_grad()
def evaluate(model, loader, num_classes, feature_size, max_ring):
    model.eval()
    correct = 0
    total = 0
    for data in loader:
        features, adj, labels, complex_batch, batch = process_tu_dataset(
            data, num_classes, feature_size, max_ring=max_ring
        )
        logits = model(features, adj, complex_batch, batch=batch)
        pred = logits.argmax(dim=-1)
        correct += int(pred.eq(labels).sum().item())
        total += int(labels.numel())
    return {"acc": correct / max(total, 1), "total": total}


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="BZR")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--hop", type=int, default=2)
    parser.add_argument("--max_ring", type=int, default=10)
    parser.add_argument("--fusion_gamma", type=float, default=0.2)
    parser.add_argument("--ring_weight", type=float, default=0.05)
    parser.add_argument("--retrieval_weight", type=float, default=0.1)
    parser.add_argument("--memory_utility_weight", type=float, default=0.1)
    parser.add_argument("--use_diff_lifting", action="store_true")
    parser.add_argument("--use_task_rerank", action="store_true")
    parser.add_argument("--use_memory_reflection", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
