# 任务模型数据说明

## 数据位置

```text
task_models/dagbench/   84份DAGBench增强任务图
task_models/stg/       180份STG约束保留任务图
task_models/synthetic/ 32份复杂合成任务图
task_models/manifest.json
```

`manifest.json` 中每个条目记录文件、来源、工作流家族、用途、增强后SHA-256、
原始文件SHA-256、任务数和依赖边数。模型应按清单读取，不应依赖目录遍历顺序。

## 单个文件结构

```json
{
  "format_version": 1,
  "kind": "enhanced_task_graph",
  "id": "dagbench:mec.sleipnir_facebook",
  "source": "dagbench",
  "family": "mec",
  "seed": 123,
  "units": {
    "workload": "source_cost_unit",
    "memory_required": "MiB",
    "edge_data": "MB"
  },
  "origin": {
    "path": "outputs/datasets/dagbench/.../graph.json",
    "raw_sha256": "..."
  },
  "tasks": [],
  "edges": []
}
```

## tasks字段

| 字段 | 含义 |
| --- | --- |
| `id` | 从0开始的任务编号 |
| `workload` | 任务基础计算量 |
| `cpu_cores_required` | 最低CPU核数需求 |
| `memory_required` | 最低内存需求，MiB |
| `accelerator_required` | 是否必须有GPU/加速器 |
| `required_features` | 资源必须具备的能力标签 |
| `task_type` | `cpu`、`data`、`gpu`或`generic` |

## edges字段

| 字段 | 含义 |
| --- | --- |
| `source` | 前驱任务ID |
| `target` | 后继任务ID |
| `data` | 前驱向后继传输的数据量，MB |

TaskGNN使用任务属性以及前驱、后继方向进行消息传播。它输出的是每个任务的
上下文向量，不是完整任务队列。任务优先级最终由Actor对当前合法动作打分得到。

## role字段

- `train_pool`：允许进入训练任务池；
- `id_validation_pool`：同分布验证候选，不参与参数更新；
- `dag_ood_pool`：保留工作流家族，用于DAG泛化测试；
- `reserved_external_test`：STG官方测试数据，不能用于训练或选模型。

