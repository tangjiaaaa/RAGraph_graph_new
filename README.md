# ReTAG 落地方案包

这个文件夹是给学生使用的轻量交付包，只包含改造 ReTAG/RAGraph 所需的核心代码和说明，不包含原项目中的数据集、模型权重、日志和无关模型。

## 1. 包内容

```text
ReTAG_落地方案包
├─ README.md
├─ code_to_copy
│  └─ RAGraph_graph_new
│     ├─ RAGraph_memory.py
│     └─ ragraph_utils
│        ├─ DiffLiftRingSelector.py
│        ├─ TaskAwareRetriever.py
│        ├─ ToyGraphBase.py
│        └─ __init__.py
├─ docs
│  ├─ RAGraph原模型集成说明_可运行版.md
│  └─ ReTAG模型优化step_by_step实验方案.md
├─ references
│  ├─ papers
│  │  └─ 借鉴论文清单与阅读顺序.md
│  └─ code
│     ├─ 代码来源与借鉴关系.md
│     ├─ difflifting
│     ├─ topobench
│     └─ ragraph_original
│  └─ full_repos
│     ├─ topobench-main
│     └─ difflifting-main
└─ scripts
   ├─ install_to_ragraph.ps1
   └─ train_memory_template.py
```

## 2. 目标原项目

默认目标项目路径：

```text
F:\research\重邮\汤佳\RAGraph-new
```

本包只改造图分类目录：

```text
RAGraph_graph_new
```

## 3. 一键复制代码

在 PowerShell 中运行：

```powershell
cd F:\research\重邮\汤佳\优化方案\ReTAG_落地方案包
powershell -ExecutionPolicy Bypass -File .\scripts\install_to_ragraph.ps1
```

如果原项目不在默认路径，用：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_to_ragraph.ps1 -TargetRoot "你的RAGraph-new路径"
```

## 4. 手动复制方式

把下面文件复制到原项目对应位置：

```text
code_to_copy\RAGraph_graph_new\RAGraph_memory.py
  -> RAGraph-new\RAGraph_graph_new\RAGraph_memory.py

code_to_copy\RAGraph_graph_new\ragraph_utils\DiffLiftRingSelector.py
  -> RAGraph-new\RAGraph_graph_new\ragraph_utils\DiffLiftRingSelector.py

code_to_copy\RAGraph_graph_new\ragraph_utils\TaskAwareRetriever.py
  -> RAGraph-new\RAGraph_graph_new\ragraph_utils\TaskAwareRetriever.py

code_to_copy\RAGraph_graph_new\ragraph_utils\ToyGraphBase.py
  -> RAGraph-new\RAGraph_graph_new\ragraph_utils\ToyGraphBase.py

code_to_copy\RAGraph_graph_new\ragraph_utils\__init__.py
  -> RAGraph-new\RAGraph_graph_new\ragraph_utils\__init__.py
```

## 5. 如何启用新模型

在原训练脚本中，把：

```python
from RAGraph2 import RAGraph
```

或：

```python
from RAGraph import RAGraph
```

改为：

```python
from RAGraph_memory import RAGraph
```

然后构造模型时可以使用：

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

## 6. 建议实验顺序

1. `S0`：原始 `RAGraph2.py`，作为 baseline。
2. `S1`：`RAGraph_memory.py`，`use_diff_lifting=False, use_task_rerank=True, use_memory_reflection=False`。
3. `S2`：`use_diff_lifting=True, use_task_rerank=False, use_memory_reflection=False`。
4. `S3`：`use_diff_lifting=True, use_task_rerank=True, use_memory_reflection=False`。
5. `S4`：`use_diff_lifting=True, use_task_rerank=True, use_memory_reflection=True`。

## 6.5 借鉴论文和参考代码

相关论文清单在：

```text
references\papers\借鉴论文清单与阅读顺序.md
```

相关 reference 代码快照在：

```text
references\code\
```

完整 reference 代码库在：

```text
references\full_repos\topobench-main
references\full_repos\difflifting-main
```

这两个目录只作为论文和代码追溯材料，不需要整体复制到 RAGraph 项目里。

其中：

```text
references\code\difflifting
```

用于说明 `DiffLiftRingSelector.py` 的来源。

```text
references\code\topobench
```

用于说明 TopoTune/Hasse route reasoning 和 GPS 消融的来源。

```text
references\code\ragraph_original
```

用于对照原 RAGraph2 和 ring feature 实现。

## 7. 反思机制的准确说法

当前实现优化的是 memory 的使用优先级，不是重写 memory embedding。

可以写：

```text
supervised reflective utility calibration for cellular memories
```

不要写：

```text
agent memory rewriting
```

## 8. 首次运行前检查

切到你平时能运行 RAGraph 的 PyTorch 环境，然后执行：

```powershell
cd F:\research\重邮\汤佳\RAGraph-new\RAGraph_graph_new
python -m py_compile RAGraph_memory.py ragraph_utils\DiffLiftRingSelector.py ragraph_utils\TaskAwareRetriever.py ragraph_utils\ToyGraphBase.py
```

如果这里通过，再运行训练脚本。
