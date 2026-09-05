# TriSched 增强数据集 v1

本目录是可以随 GitHub 仓库分发的完整增强数据集。任务图与资源图分别保存，
训练或评测时再由场景装配器组合，因此同一个任务DAG可以在不同云边端网络上
运行，同一个资源网络也可以承载不同工作流。

## 目录

```text
data/enhanced-v1/
├── manifest.json
├── task_models/
│   ├── manifest.json
│   ├── dagbench/
│   ├── stg/
│   └── synthetic/
├── resource_models/
│   ├── manifest.json
│   ├── dagbench/
│   ├── topology_zoo/
│   └── synthetic/
├── TASK_DATASET.md
├── RESOURCE_DATASET.md
└── INTEGRATION.md
```

当前包含：

- 296份任务图：DAGBench 84、STG 180、复杂合成32；
- 152份资源图：DAGBench网络84、Topology Zoo 36、复杂合成32。

所有路径均相对于本目录或项目根目录，不包含生成者电脑的绝对路径。

## 两套模型如何使用

- `TriSchedGNNPPOPolicy`：重点使用任务图消息传递；资源属性仍通过当前合法
  `(task, resource)` 候选特征进入Actor。
- `CloudEdgeDualGraphPolicy`：同时使用任务图编码和资源图编码，再为合法
  `(task, resource)` 组合打分。

两种模型都不能只读取任务文件就完成调度。环境在运行前必须把一份任务图和
一份资源图组合为完整 `Scenario`，然后逐步生成合法动作集合。

## 直接运行

仓库中的正式配置优先读取本目录，因此其他同学克隆后无需重新下载原始数据：

```powershell
python -m trisched materialize-mixed-dataset --config configs/complex_dual_graph.json
python -m trisched train-dual-graph --config configs/complex_dual_graph.json
```

只有重新生成或审计增强数据时，才需要：

```powershell
.\scripts\fetch_mixed_datasets.ps1
python -m trisched export-enhanced-datasets `
  --config configs/complex_dual_graph.json `
  --output data/enhanced-v1
```

不要手工修改单个JSON。修改增强规则后应重新运行导出命令，使文件内容、清单
哈希和划分角色保持一致。

