# RAGraph 原模型集成说明：DiffLift + Task-Aware Retrieval + Reflective Cellular Memory

日期：2026-06-17  
目标代码目录：`F:\research\重邮\汤佳\RAGraph-new\RAGraph_graph_new`

这版不是独立草案，而是已经写进你的原始模型代码目录，并保持原 `RAGraph2.py` 的调用接口。

## 1. 已经写入/修改的文件

### 1.1 新增：可学习 2-cell selector

文件：

```text
F:\research\重邮\汤佳\RAGraph-new\RAGraph_graph_new\ragraph_utils\DiffLiftRingSelector.py
```

作用：

- 读取你原本 `Complex` 里的 `cochains[2].boundary_index`。
- 不重新构造 complex。
- 只在已有候选 ring / 2-cell 上学习选择权重。
- 对应 DiffLifting 的思想：从 node embedding 出发，学习候选高阶 cell 是否应该被纳入。

在模型中的使用位置：

```python
ring_mean, selector_loss, info = self.ring_selector(
    node_emb,
    complex_obj.cochains[2].boundary_index.to(node_emb.device),
    self._edge_boundary_from_complex(complex_obj).to(node_emb.device),
)
```

替换的原始逻辑：

```python
single_loss, ring_mean = ring_contrastive_loss(...)
```

### 1.2 新增：Task-aware reranker

文件：

```text
F:\research\重邮\汤佳\RAGraph-new\RAGraph_graph_new\ragraph_utils\TaskAwareRetriever.py
```

作用：

- 输入 query key：`concat(graph_emb, ring_mean)`。
- 输入 topK retrieved values：`rag_embeddings`。
- 基于 query-candidate 交互重新生成 `rag_weights`。
- 用训练标签构造弱监督 retrieval alignment loss。

在模型中的使用位置：

```python
rag_weights, utility_logits = self.reranker(query_keys, rag_embeddings, rag_weights)
retrieval_loss = retrieval_alignment_loss(utility_logits, rag_labels, label)
```

替换/增强的原始逻辑：

```python
rag_embedding = torch.sum(rag_weights.unsqueeze(-1) * rag_embeddings, dim=1)
```

原来 `rag_weights` 只来自固定相似度；现在先固定相似度召回，再 task-aware rerank。

### 1.3 修改：`ragraph_utils/__init__.py`

文件：

```text
F:\research\重邮\汤佳\RAGraph-new\RAGraph_graph_new\ragraph_utils\__init__.py
```

新增导出：

```python
from .DiffLiftRingSelector import DiffLiftRingSelector
from .TaskAwareRetriever import TaskAwareReranker, retrieval_alignment_loss
```

作用：

让新模型可以继续沿用：

```python
from ragraph_utils import DiffLiftRingSelector, TaskAwareReranker
```

### 1.4 修改：`ToyGraphBase.py`

文件：

```text
F:\research\重邮\汤佳\RAGraph-new\RAGraph_graph_new\ragraph_utils\ToyGraphBase.py
```

修改 1：新增 memory utility 状态。

```python
self.utility_weight = 0.0
self.utility_momentum = 0.9
self.resource_utility = None
```

在 `build_toy_graph` 结束后初始化：

```python
self.resource_utility = torch.zeros(self.resource_keys.size(0), device=self.resource_keys.device)
```

修改 2：`retrieve` 兼容返回 topK indices。

新接口：

```python
retrieve(..., return_indices=False)
```

老代码不受影响：

```python
rag_embeddings, rag_labels, rag_weights = retrieve(...)
```

新模型使用：

```python
rag_embeddings, rag_labels, rag_weights, topk_indices = retrieve(..., return_indices=True)
```

修改 3：训练时更新 memory utility。

新增函数：

```python
update_memory_utility(topk_indices, target_labels, rag_weights)
```

其含义是：

- retrieved memory 的 label 和当前训练样本 label 一致：utility 上升。
- 不一致：utility 下降。
- 更新幅度乘以当前 rerank 后的检索权重。
- 动量更新，避免一次 batch 剧烈改变 memory。

### 1.5 新增：真正接入原模型的新 RAGraph

文件：

```text
F:\research\重邮\汤佳\RAGraph-new\RAGraph_graph_new\RAGraph_memory.py
```

这是当前应该实验的主文件。

它保持和 `RAGraph2.py` 基本一致的初始化和 forward 接口：

```python
rag_model = RAGraph(
    pretrain_model,
    resource_dataset=train_val_dataset,
    feture_size=feature_size,
    num_class=num_classes,
    emb_size=hid_units,
    finetune=True,
)
```

新增开关：

```python
use_diff_lifting=True
use_task_rerank=True
use_memory_reflection=True
memory_utility_weight=0.1
retrieval_weight=0.1
ring_weight=0.05
```

## 2. 训练脚本如何切换

如果你的训练脚本原来是：

```python
from RAGraph2 import RAGraph
```

或者：

```python
from RAGraph import RAGraph
```

改成：

```python
from RAGraph_memory import RAGraph
```

其他构造参数可以先不改。为了做消融，建议显式写：

```python
rag_model = RAGraph(
    pretrain_model,
    resource_dataset=train_val_dataset,
    feture_size=feature_size,
    num_class=num_classes,
    emb_size=hid_units,
    finetune=True,
    ring_weight=0.05,
    retrieval_weight=0.1,
    query_graph_hop=2,
    retrieve_num=5,
    fusion_gamma=0.2,
    max_ring=10,
    use_diff_lifting=True,
    use_task_rerank=True,
    use_memory_reflection=True,
    memory_utility_weight=0.1,
).cuda()
```

## 3. 实验步骤和对应代码位置

## Step 0：原始 RAGraph2 baseline

代码位置：

```text
F:\research\重邮\汤佳\RAGraph-new\RAGraph_graph_new\RAGraph2.py
```

训练脚本 import：

```python
from RAGraph2 import RAGraph
```

目的：

确认原始 ReTAG/RAGraph2 的结果。

## Step 1：只启用 task-aware rerank，不启用 learned lifting 和 reflection

代码位置：

```text
RAGraph_memory.py
TaskAwareRetriever.py
```

训练脚本 import：

```python
from RAGraph_memory import RAGraph
```

模型参数：

```python
use_diff_lifting=False
use_task_rerank=True
use_memory_reflection=False
```

验证问题：

固定相似度 topK 后，加 query-candidate rerank 是否提升。

## Step 2：启用 DiffLift-style learned ring selector

代码位置：

```text
RAGraph_memory.py
DiffLiftRingSelector.py
```

模型参数：

```python
use_diff_lifting=True
use_task_rerank=False
use_memory_reflection=False
```

验证问题：

学习 ring/2-cell 选择是否优于静态 `ring_mean`。

## Step 3：DiffLift selector + task-aware rerank

代码位置：

```text
RAGraph_memory.py
DiffLiftRingSelector.py
TaskAwareRetriever.py
```

模型参数：

```python
use_diff_lifting=True
use_task_rerank=True
use_memory_reflection=False
```

验证问题：

“学 cellular key” 和 “学检索权重” 是否互补。

## Step 4：启用 reflective cellular memory

代码位置：

```text
RAGraph_memory.py
ToyGraphBase.py
```

模型参数：

```python
use_diff_lifting=True
use_task_rerank=True
use_memory_reflection=True
memory_utility_weight=0.1
```

验证问题：

训练过程中是否能逐渐提高有用 memory 的检索概率，降低 harmful memory 的影响。

## 4. 当前“反思”机制是否真的能优化记忆内容？

可以，但要准确理解它的能力边界。

### 4.1 它现在能优化什么

当前实现优化的是：

> memory item 的使用优先级，而不是 memory item 的 embedding 内容本身。

也就是说：

- `resource_keys` 不会被改。
- `resource_values` 不会被改。
- `resource_labels` 不会被改。
- 会更新 `resource_utility`。

检索分数变为：

```text
score = structure_score + semantic_score + ring_score + utility_weight * resource_utility
```

因此它能让训练中反复证明有帮助的 memory 更容易被检索到，让经常误导模型的 memory 降权。

### 4.2 为什么它是有效的

你的任务是监督图分类。每个训练 query 有标签，资源库 memory 也有标签。  
如果一个 memory 经常在相似 query 中和标签一致，它大概率是有用检索；如果经常不一致，它就是 harmful retrieval。

当前更新规则：

```text
label match: utility += positive update
label mismatch: utility += negative update
update magnitude proportional to rerank weight
```

这比原始固定相似度更合理，因为原始检索只知道“像不像”，不知道“对当前任务有没有帮助”。

### 4.3 它不能声称什么

不能说：

```text
The model autonomously reflects like an agent.
```

也不能说：

```text
The memory content is rewritten.
```

更准确的论文表述是：

```text
We introduce a supervised reflective utility update for cellular memories, which calibrates the retrieval priority of each memory according to downstream task feedback.
```

中文理解：

> 这是“反思式 memory 使用权重管理”，不是 agent 式自主记忆重写。

### 4.4 如果想让反思更强，下一版应该怎么做

当前是 V1：更新 memory utility。

V2 可以做：更新 memory prototype。

```python
resource_keys[i] = momentum * resource_keys[i] + (1 - momentum) * query_key
resource_ring_feats[i] = momentum * resource_ring_feats[i] + (1 - momentum) * query_ring
```

但这一步风险更大，可能造成 label leakage 或 prototype collapse，所以建议先跑 V1。

## 5. 推荐主实验顺序

| 实验名 | 文件 | 参数 |
|---|---|---|
| S0 | `RAGraph2.py` | 原始 baseline |
| S1 | `RAGraph_memory.py` | `diff=False, rerank=True, reflect=False` |
| S2 | `RAGraph_memory.py` | `diff=True, rerank=False, reflect=False` |
| S3 | `RAGraph_memory.py` | `diff=True, rerank=True, reflect=False` |
| S4 | `RAGraph_memory.py` | `diff=True, rerank=True, reflect=True` |

建议先不要加 Hasse/Transformer。先验证这条主线：

```text
静态 cellular retrieval
-> learned cellular retrieval key
-> task-aware memory rerank
-> reflective utility calibration
```

这条线和你的论文 ReTAG 叙事最贴合，也最容易解释增益来源。

## 6. 已做的基本检查

已通过 Python 编译检查：

```powershell
python -m py_compile `
  F:\research\重邮\汤佳\RAGraph-new\RAGraph_graph_new\RAGraph_memory.py `
  F:\research\重邮\汤佳\RAGraph-new\RAGraph_graph_new\ragraph_utils\DiffLiftRingSelector.py `
  F:\research\重邮\汤佳\RAGraph-new\RAGraph_graph_new\ragraph_utils\TaskAwareRetriever.py `
  F:\research\重邮\汤佳\RAGraph-new\RAGraph_graph_new\ragraph_utils\ToyGraphBase.py
```

尚未完成：

- 尚未跑完整训练。
- 尚未验证每个数据集的性能。
- 尚未把 `vanilla-rag.py / pretrain3.py / finetune` 脚本批量改成新模型。

下一步应该选一个你当前最常用的训练脚本，把 import 改成 `RAGraph_memory`，先跑 ENZYMES 或 BZR 的单 seed smoke test。
