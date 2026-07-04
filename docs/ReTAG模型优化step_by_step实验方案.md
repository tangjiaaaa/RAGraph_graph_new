# ReTAG / RAGraph 模型优化 Step-by-Step 实验方案

生成日期：2026-06-17  
主论文草稿：`F:\research\重邮\汤佳\优化方案\TKDE_1223.pdf`  
原始代码：`F:\research\重邮\汤佳\RAGraph-new`  
TopoBench 参考：`F:\research\重邮\汤佳\优化方案\ref-codes\topobench-main\topobench-main`  
DiffLifting 参考：`F:\research\重邮\汤佳\优化方案\ref-codes\difflifting-main\difflifting-main`

## 0. 结论先行

你的当前论文叙事是 **Retrieved Cellular Topologies-Augmented Graph Learning / ReTAG**，核心是：

1. 把图 lift 到 cell complex。
2. 从资源库检索语义和拓扑相似的 cellular topology。
3. 用多维拓扑消息传递和 cellular contrastive learning 增强预测。

结合 TopoTune 和 DiffLifting 后，最合理的优化方向不是简单替换成 Graph Transformer，而是：

> 从“静态 ring 均值 + 固定权重检索”升级为“可学习 cellular lifting + task-aware retrieval + reflective cellular memory 管理”。

建议最终方法名可以写成：

> **ReTAG++: Reflective Cellular Memory for Retrieval-Augmented Graph Learning**

或更保守：

> **Task-Aware Cellular Memory for Retrieved Cellular Topologies-Augmented Graph Learning**

注意：这里的 memory 不是 agent，而是 graph learning 中的外部 cellular memory。可以写成 “inspired by reflective memory management”，不要直接宣称 agent memory。

## 1. 代码和论文对齐后的问题诊断

### 1.1 你的当前 RAGraph 实现

关键文件：

- `F:\research\重邮\汤佳\RAGraph-new\RAGraph.py`
- `F:\research\重邮\汤佳\RAGraph-new\RAGraph_graph_new\RAGraph.py`
- `F:\research\重邮\汤佳\RAGraph-new\RAGraph_graph_new\RAGraph2.py`
- `F:\research\重邮\汤佳\RAGraph-new\RAGraph_graph_new\ragraph_utils\ToyGraphBase.py`
- `F:\research\重邮\汤佳\RAGraph-new\RAGraph_graph_new\ragraph_utils\extract_ring.py`

当前主流程是：

```text
node_emb = pretrain_model.embed(features, adj)
graph_emb = mean_pool(node_emb)
ring_mean = ring_contrastive_loss(...)[1]
query_key = concat(graph_emb, ring_mean)
retrieved = ToyGraphBase.retrieve(query_key, structure_signature, ring_mean)
logits = decoder(concat(query_emb_pool, rag_embedding, ring_feat))
```

主要短板：

1. `ring_mean` 是静态候选 ring 的均值，无法学习哪些 2-cell 对任务有用。
2. `ToyGraphBase.retrieve` 里 `structure_weight / semantic_weight / ring_weight` 是固定权重。
3. 检索 topK 后直接 softmax 聚合，缺少 query-candidate 交互式 rerank。
4. memory/knowledge base 是一次性构建，缺少基于训练反馈的更新、降权、遗忘机制。
5. 论文中写了 multi-dimensional topology reasoning，但代码中主要是 ring feature + retrieval fusion，真正的 rank-aware 传播还不够强。

### 1.2 TopoTune 给你的启发

TopoTune 关键文件：

- `topobench\nn\backbones\combinatorial\gccn.py`
- `topobench\nn\backbones\graph\gps.py`
- `configs\model\cell\topotune.yaml`

TopoTune 的核心不是“直接在 cell complex 上做 Transformer”，而是：

1. 把不同 rank 的 cell 关系展开成 Hasse graph / route graph。
2. 对每条 route 复用 GCN/GIN/GAT 等基础 GNN。
3. 对不同 rank 的输出做聚合更新。

因此对你最有价值的是：

- 用 rank-aware Hasse graph 表示 cell memory。
- 对 0/1/2-cell 的信息分开建模，而不是只池化成一个 `ring_mean`。
- Transformer/GPS 只作为一个可控消融，不应该作为默认主方法。

### 1.3 DiffLifting 给你的启发

DiffLifting 关键文件：

- `model\tnn_with_lifting_graph_classific.py`
- `model\tnn_with_lifiting.py`
- `model\TNN.py`

DiffLifting 的核心是：

1. 先用 GIN/GPS 得到 node latent embedding。
2. 学习 `k_v` 或候选高阶结构大小。
3. 用 MLP 输出候选 edge/cell 的纳入概率。
4. 用 Gumbel-softmax 或 STE 构造可反传的 incidence。
5. 把 learned lifting 后的 complex 交给 CWN/SCN2/CXN/TopoTune 等 TNN。

你的接入点：

> 不要一开始重写 incidence 全流程。先对已有 ring/cycle candidates 学一个 selector，把 `ring_mean` 改成 differentiable selected cellular representation。

## 2. 总体优化路线

建议按 6 个版本逐步实验，每一步都可以单独发 ablation。

| 步骤 | 核心改动 | 名称 | 预期收益 |
|---|---|---|---|
| S0 | 复现实验基线 | Original RAGraph | 建立可靠 baseline |
| S1 | 把 toy graph base 显式改成 memory bank | Explicit Cellular Memory | 为动态管理做接口 |
| S2 | 学习选择任务相关 2-cell | DiffLift Ring Selector | 提高 cellular topology 表示质量 |
| S3 | 对 topK memory 做 query-candidate rerank | Task-Aware Reranker | 降低错误检索影响 |
| S4 | 用任务反馈更新 memory utility | Reflective Cellular Memory | 支撑 reflective memory 叙事 |
| S5 | 引入 rank-aware Hasse reasoning，可消融 GCN vs Transformer | Hasse/TopoTune Encoder | 验证 Transformer 是否真的有用 |

对应代码已写到当前目录：

- `retag_step_code/step1_cellular_memory_bank.py`
- `retag_step_code/step2_diff_lifting_ring_selector.py`
- `retag_step_code/step3_task_aware_retriever.py`
- `retag_step_code/step4_hasse_topotune_adapter.py`
- `retag_step_code/step5_memory_ragraph_integration.py`
- `retag_step_code/step6_experiment_runner_template.py`

## 3. Step-by-Step 实验方案

## Step 0：复现并固定当前强 baseline

目标：先确认当前 ReTAG/RAGraph 在 PROTEINS、ENZYMES、BZR、COX2 上的稳定结果。

建议固定：

- 5 个随机种子。
- `topK = 5`
- `query_graph_hop = 1 / 2`
- `max_ring = 6 / 8 / 10`
- 是否使用 `ring_loss`
- 是否使用 `RAGraph.py` 或 `RAGraph2.py`

需要记录：

```text
dataset, seed, topK, hopK, max_ring, ring_weight, accuracy/f1/auc, std
```

最小命令示例：

```powershell
cd F:\research\重邮\汤佳\RAGraph-new\RAGraph_graph_new
python pretrain3.py
```

注意：你现在的 `RAGraph2.py` 已经比 `RAGraph.py` 更接近论文，因为它把 `ring_loss` 加进了训练，并增加了 `fusion_gamma`、`graph_decoder`。后续建议以 `RAGraph2.py` 作为主分支。

## Step 1：显式 Cellular Memory Bank

动机：你论文里说 knowledge base / cellular topology memory，但代码中 `ToyGraphBase` 更像静态 tensor cache。先把 memory 抽象成显式模块，后续才能写 update / forget / reflect。

对应代码：

```text
retag_step_code/step1_cellular_memory_bank.py
```

接入方式：

1. 保留 `ToyGraphBase.build_toy_graph` 的特征构建逻辑。
2. 把 `resource_keys / resource_values / resource_labels / resource_ring_feats` 写入 `CellularMemoryBank.add(...)`。
3. 在 `retrieve` 阶段调用：

```python
values, labels, weights, indices = memory_bank.retrieve(
    query_keys=query_keys,
    query_cell_feats=ring_mean,
    topk=5,
)
```

这一版不要引入复杂学习，只改变工程结构。

实验：

- S0: 原 `ToyGraphBase.retrieve`
- S1: `CellularMemoryBank.retrieve`

如果 S1 和 S0 结果接近，说明重构没有破坏 baseline。

## Step 2：DiffLift-style 可学习 Ring Selector

动机：DiffLifting 的关键不是“多一种静态 lifting”，而是从 latent representation 中学习高阶 cell 是否应被纳入。你这里可以先从 ring candidates 选择开始。

对应代码：

```text
retag_step_code/step2_diff_lifting_ring_selector.py
```

核心替换：

原来：

```python
single_loss, ring_mean = ring_contrastive_loss(
    node_emb,
    complex_obj.cochains[2].boundary_index,
    edge_index,
)
```

替换为：

```python
ring_info = self.ring_selector.forward_from_boundary(
    node_emb,
    complex_obj.cochains[2].boundary_index.to(node_emb.device),
    edge_index.to(node_emb.device),
)
ring_mean = ring_info["ring_mean"]
ring_aux_loss = ring_info["aux_loss"]
```

训练 loss：

```python
total_loss = cls_loss + lambda_ring * ring_loss + lambda_sparse * ring_aux_loss
```

推荐超参：

```text
lambda_ring = 0.03, 0.05, 0.1
temperature = 0.5, 1.0
hard = false 先跑，稳定后再 hard=true
```

实验：

- S1: static ring mean
- S2-a: learned soft ring selector
- S2-b: learned hard/STE ring selector

论文表述：

> Inspired by differentiable lifting, we learn a task-adaptive distribution over candidate 2-cells and aggregate cellular representations with differentiable selection weights.

## Step 3：Task-Aware Retrieval Reranker

动机：Task-Aware Retrieval Augmentation 的价值不在于“用了 transformer”，而在于它显式判断 retrieved item 是否对当前 task/query 有帮助。你当前 topK 只靠固定相似度，容易把拓扑相似但标签无关的 memory 拉进来。

对应代码：

```text
retag_step_code/step3_task_aware_retriever.py
```

接入位置：

```text
ToyGraphBase.retrieve -> 得到 topK candidates -> TaskAwareReranker -> 新 weights -> 聚合
```

核心逻辑：

```python
query = torch.cat([graph_emb, ring_mean], dim=-1)
rerank_weights, utility_logits = self.reranker(
    query=query,
    candidates=rag_embeddings,
    base_weights=rag_weights,
    candidate_labels=rag_labels,
)
rag_embedding = torch.sum(rerank_weights.unsqueeze(-1) * rag_embeddings, dim=1)
```

如果训练集有 label，可以加弱监督：

```python
retrieval_loss = retrieval_supervision_loss(utility_logits, rag_labels, label)
total_loss = cls_loss + lambda_ret * retrieval_loss
```

推荐超参：

```text
lambda_ret = 0.05, 0.1, 0.2
```

实验：

- S2: no rerank
- S3-a: MLP rerank
- S3-b: MLP rerank + retrieval supervision

论文表述：

> We introduce a task-aware cellular reranker that estimates the utility of each retrieved cellular memory conditioned on the query graph and its selected 2-cell representation.

## Step 4：Reflective Cellular Memory 管理

动机：如果你想讲 memory，而不是普通 RAG，就必须有持续管理机制。最小可行版本是：每次训练后估计 memory 是否有帮助，然后更新 utility，推理时把 utility 当作 retrieval bias。

对应代码：

```text
retag_step_code/step1_cellular_memory_bank.py
```

反思信号建议：

```text
utility_delta = loss_without_memory - loss_with_memory
```

也可以简化为：

```text
预测正确且候选 label 匹配：+1
预测错误且候选 label 不匹配：-1
否则：0
```

调用方式：

```python
memory_bank.reflect(
    retrieved_indices=topk_indices,
    utility_delta=utility_delta,
)
```

检索时：

```python
score = semantic_score + cell_score + utility_weight * memory_utility
```

推荐超参：

```text
utility_momentum = 0.9
utility_weight = 0.05, 0.1, 0.2
max_memory_size = 5000 / 10000 / all
```

实验：

- S3: static memory
- S4-a: memory utility update
- S4-b: utility update + pruning
- S4-c: utility update + harmful memory downweight only

论文表述：

> Unlike agent memory for autonomous decision-making, our memory is designed for graph representation learning. It reflectively updates the utility of cellular memories according to downstream prediction feedback.

这个说法比较稳，不会被审稿人质疑“你不是 agent”。

## Step 5：TopoTune/Hasse-style Rank-Aware Reasoning

动机：你论文里写 multi-dimensional topological message passing，但当前实现更像 “ring feature + RAG fusion”。可以增加一个轻量 Hasse encoder，让 0/1/2-cell 真正做 rank-aware propagation。

对应代码：

```text
retag_step_code/step4_hasse_topotune_adapter.py
```

接入方式：

```python
hasse_data = complex_to_hasse_data(
    x0=node_emb,
    edge_index=edge_index,
    incidence_2=complex_obj.cochains[2].boundary_index,
)
hasse_emb = hasse_encoder(hasse_data)
```

然后替换或拼接：

```python
graph_emb = graph_emb + alpha * hasse_emb
```

或：

```python
decoder_input = torch.cat([graph_emb, hasse_emb, rag_embedding, ring_feat], dim=-1)
```

消融重点：

- Hasse-GCN 是否提升。
- Hasse-Transformer 是否提升。
- Transformer 是否只是增加参数而非带来稳定收益。

建议实验：

| 实验 | Encoder | 目的 |
|---|---|---|
| S5-a | no Hasse | 对照 |
| S5-b | Hasse-GCN | 验证 rank-aware cell propagation |
| S5-c | Hasse-TransformerConv | 验证 transformer 是否必要 |
| S5-d | Hasse-GCN + DiffLift selector | 验证 learned lifting 和 rank-aware propagation 是否互补 |

论文表述：

> Following the route-expansion view of generalized combinatorial complex neural networks, we construct a lightweight Hasse graph over 0-, 1-, and 2-cells to perform rank-aware propagation.

不要写：

> We use a cellular graph transformer and therefore outperform GNNs.

更稳妥的是：

> We evaluate both message-passing and attention-based Hasse encoders, and empirically find that task-aware cellular memory contributes more consistently than replacing local propagation with global attention.

## Step 6：集成版 ReTAG++ Head

对应代码：

```text
retag_step_code/step5_memory_ragraph_integration.py
```

这个文件提供 `MemoryRAGraphHead`，它把三件事合在一起：

1. `DiffLiftRingSelector`
2. `TaskAwareReranker`
3. `ring/retrieval auxiliary loss`

推荐最小集成方式：

1. 保留你的 `pretrain_model.embed(features, adj)`。
2. 保留 `ToyGraphBase.retrieve(...)` 得到初始 topK。
3. 把 topK 结果交给 `MemoryRAGraphHead`。

伪代码：

```python
node_emb, _ = pretrain_model.embed(features, adj)
graph_emb = graph_pool(node_emb, batch)

query_key = torch.cat([graph_emb, static_or_learned_ring_mean], dim=-1)
rag_embeddings, rag_labels, rag_weights = toy_graph_base.retrieve(...)

out = memory_head(
    node_emb=node_emb,
    graph_emb=graph_emb,
    complex_batch=complex_batch,
    memory_values=rag_embeddings,
    memory_labels=rag_labels,
    memory_weights=rag_weights,
    labels=label,
    batch=batch,
)
loss = out["loss"]
logits = out["logits"]
```

## 4. 实验矩阵

建议主表：

| Model | Learned 2-cell | Task rerank | Reflective memory | Hasse reasoning | PROTEINS | ENZYMES | BZR | COX2 |
|---|---|---|---|---|---:|---:|---:|---:|
| GCN/GIN backbone | no | no | no | no | | | | |
| RAGraph | no | no | no | no | | | | |
| ReTAG-S1 | no | no | explicit only | no | | | | |
| ReTAG-S2 | yes | no | no | no | | | | |
| ReTAG-S3 | yes | yes | no | no | | | | |
| ReTAG-S4 | yes | yes | yes | no | | | | |
| ReTAG-S5-GCN | yes | yes | yes | Hasse-GCN | | | | |
| ReTAG-S5-TF | yes | yes | yes | Hasse-Transformer | | | | |

必须加的消融：

| 消融 | 说明 |
|---|---|
| w/o cellular memory | 去掉 ring/cell memory，只用 graph RAG |
| w/o learned lifting | 用静态 ring mean |
| w/o task-aware rerank | 只用固定相似度 |
| w/o reflection | 不更新 utility |
| w/o ring contrastive | 不加 CTCL |
| GCN vs Transformer | 验证 Transformer 是否必要 |

## 5. 预期结果和论文解释

最可能出现的结果：

1. `S2` 比 `S1` 稳定提升，因为 learned selector 会抑制无用 ring。
2. `S3` 在 few-shot / distribution shift 下提升更明显，因为 rerank 会减少 harmful retrieval。
3. `S4` 对动态图或多轮训练更有叙事价值，但静态 TU 数据集上提升可能有限。
4. `S5-Transformer` 不一定优于 `S5-GCN`，这不是坏事。你可以把结论写成：cellular memory 和 learned lifting 是主要收益来源，Transformer 只是可选增强。

建议论文中的核心创新改写为：

> We propose a reflective cellular memory mechanism for retrieval-augmented graph learning. The memory stores graph-level and 2-cell-level topology representations, learns task-adaptive cellular lifting over candidate cycles, reranks retrieved memories according to query-specific utility, and reflectively updates memory utility using downstream feedback.

## 6. 推荐实施顺序

### 第 1 周：稳定 baseline

1. 用 `RAGraph2.py` 跑 PROTEINS/ENZYMES/BZR/COX2。
2. 固定 seeds。
3. 得到 S0 表格。

### 第 2 周：S1 + S2

1. 接入 `CellularMemoryBank`。
2. 接入 `DiffLiftRingSelector`。
3. 跑 `soft selector`。
4. 再跑 `hard/STE selector`。

### 第 3 周：S3

1. 接入 `TaskAwareReranker`。
2. 先不加 retrieval supervision。
3. 再加 weak retrieval supervision。
4. 统计 topK 中 label match ratio 和 rerank 后权重分布。

### 第 4 周：S4

1. 实现 utility update。
2. 做有无 pruning 的 ablation。
3. 在动态边预测或 time split 数据上重点展示。

### 第 5 周：S5

1. 接入 Hasse-GCN。
2. 接入 Hasse-Transformer。
3. 重点对比是否值得在主模型里保留 Transformer。

## 7. 写作建议

你的论文当前已经有 “cellular topologies / knowledge base / retrieval” 的叙事。优化后建议把 related work 和 method 改成四段：

1. Retrieval-Augmented Graph Learning：RAGraph 是直接 baseline。
2. Topological Graph Learning：CWN、cell complex、TopoTune。
3. Differentiable Lifting：DiffLifting，说明静态 lifting 不一定最优。
4. Memory-Augmented Learning：强调你不是 agent，而是 task-aware cellular memory。

避免过度表述：

```text
不建议：agent memory for graph learning
建议：agent-inspired reflective memory management for graph representation learning
更建议：reflective cellular memory for retrieval-augmented graph learning
```

## 8. 下一步代码集成建议

最稳的实际集成顺序：

1. 在 `RAGraph_graph_new\ragraph_utils` 下复制：
   - `step2_diff_lifting_ring_selector.py`
   - `step3_task_aware_retriever.py`
2. 新建一个 `RAGraph_memory.py`，从 `RAGraph2.py` 复制后逐步替换。
3. 先只替换 ring 计算，不动 retrieval。
4. 再替换 topK rerank。
5. 最后再加 reflective memory。

这样每一版都有独立实验，不会把所有改动混在一起导致不知道提升来自哪里。
