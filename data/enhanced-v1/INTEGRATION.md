# 对接说明：任务图与资源图组合

## 推荐接口

```python
from trisched.mixed_dataset import (
    assemble_enhanced_graphs,
    load_enhanced_resource_graph,
    load_enhanced_task_graph,
)

task_graph = load_enhanced_task_graph("data/enhanced-v1/task_models/...json")
resource_graph = load_enhanced_resource_graph(
    "data/enhanced-v1/resource_models/...json"
)

scenario = assemble_enhanced_graphs(
    task_graph,
    resource_graph,
    scenario_id="experiment-0001",
    seed=20260904,
    max_resource_nodes=12,
)
```

得到的`scenario`与现有代码完全兼容，可以交给：

```python
from trisched.env import HeterogeneousDagEnv, run_policy
from trisched.dual_graph import CloudEdgeDualGraphPolicy

env = HeterogeneousDagEnv(scenario)
legal_actions = env.candidate_actions()
result = run_policy(scenario, CloudEdgeDualGraphPolicy())
```

## 实际决策顺序

```text
任务图文件 ─→ TaskGNN ─┐
                        ├→ Actor为合法(task, resource)打分 → Env执行
资源图文件 ─→ ResourceGNN┘
```

任务图和资源图分开保存不代表分别生成两个互不相关的最终队列。Env先根据任务
就绪状态和资源兼容性生成合法组合，Actor再同时利用两个图的上下文确定本步优先级。

## 组合检查

装配器会：

1. 校验CPU、内存、GPU和能力标签；
2. 必要时扩展一个最高能力、优先cloud的资源，保证每个任务至少有一个合法动作；
3. 按任务类型和资源类型生成显式task×resource执行时间；
4. 保留任务依赖数据量、资源带宽和时延；
5. 可选地限制过大的资源图，同时保留device、edge、cloud三层节点。

正式实验不要随意组合训练池与保留池。应使用根配置和混合场景清单生成固定的
train、ID、DAG-OOD、Network-OOD和Joint-OOD切片，避免数据泄漏。

## 清单读取

根入口为`data/enhanced-v1/manifest.json`。任务清单和资源清单中的`file`都以
增强数据集目录为基准，统一使用`/`，在Windows和Linux/openEuler上都可解析。
