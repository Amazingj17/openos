# 资源模型数据说明

## 数据位置

```text
resource_models/dagbench/      84份DAGBench自带网络
resource_models/topology_zoo/ 36份Topology Zoo真实网络骨架
resource_models/synthetic/    32份合成云边端网络
resource_models/manifest.json
```

## 单个文件结构

```json
{
  "format_version": 1,
  "kind": "enhanced_resource_graph",
  "id": "topology-zoo:Oxford",
  "source": "topology_zoo",
  "family": "oxford",
  "units": {
    "memory_capacity": "MiB",
    "bandwidth": "MB/s",
    "latency": "seconds"
  },
  "resources": [],
  "bandwidth": [],
  "latency": []
}
```

## resources字段

| 字段 | 含义 |
| --- | --- |
| `id` | 从0开始的资源编号 |
| `name` | 资源名称 |
| `kind` | 只允许`device`、`edge`、`cloud` |
| `speed` | 相对计算速度 |
| `cpu_cores` | CPU核数 |
| `memory_capacity` | 内存容量，MiB |
| `has_accelerator` | 是否具有GPU/加速器 |
| `features` | 资源能力标签 |

`bandwidth[i][j]`和`latency[i][j]`分别描述资源i到资源j的有效带宽与时延。
Topology Zoo的稀疏物理网络在增强阶段转换成节点间有效路径矩阵；运行时可以通过
`max_resource_nodes`限制计算节点数量。

ResourceGNN使用静态资源字段以及运行过程中产生的`ready_time`、
`assigned_task_ratio`传播资源上下文。资源优先级不是提前固定好的永久队列，而是
Actor在每一步根据任务、网络和当前排队状态重新计算。

## role字段

- `train_pool`：训练资源网络候选；
- `id_validation_pool`：同分布验证候选；
- `network_ood_pool`：训练阶段完全保留的Topology Zoo网络；
- `not_selected`：当前划分规则不使用，但仍保留在完整增强集合中。

