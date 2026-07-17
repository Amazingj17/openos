# P1-A03 task-GNN 设计与验收契约

- 任务：P1-A03 / P1-04
- 主责：成员 A
- 复核：成员 B
- 日期：2026-07-17
- 状态：进行中（最小前向、动作接口和 checkpoint 已完成；BC/PPO 尚未接入）
- 前置门禁：[P1-03 第二轮独立复核通过](./P1-03第二轮独立复核记录.md)

## 1. 目标与单变量边界

本任务判断 DAG 消息传递能否在相同 masked PPO 闭环中优于现有 MLP 表示。第一阶段只建立可审计的 task-GNN 前向，不报告训练收益，也不访问公开 test。

必须保持不变：

- 14 维 teacher-free 候选特征及归一化含义；
- `ready task × resource` 合法动作集合和环境状态转移；
- 增量 makespan/HEFT 奖励、`gamma=1.0` 和 GAE/PPO 公式；
- 冻结的 120/30/30 split，训练和选模只加载 train/validation；
- seeds `20260717/20260718/20260719`；
- BC warm start、validation selection key 和无改善时回退 warm start 的规则；
- MLP/GNN 使用相同训练预算、失败惩罚和评测器。

第一轮禁止同时加入 resource-GNN、注意力、课程学习、OOD 扩增、新奖励、因子化动作或额外 HEFT 决策特征。若 task-GNN 未产生固定 validation 配对优势，保留 MLP，不用第二个新变量补救。

## 2. 输入与图张量契约

每个合法 task-resource 候选继续使用 P1-A02 的 14 维输入：

```text
task_workload, execution_time, resource_speed,
earliest_start, earliest_finish, resource_ready,
communication_delay, upward_rank, indegree, outdegree,
progress, is_device, is_edge, is_cloud
```

`is_heft_task` 和 `is_heft_pair` 仍被强制删除。task-GNN 的节点输入只是上述 14 维中的四个任务局部字段：

```text
task_workload, upward_rank, indegree, outdegree
```

因此 GNN 没有获得 MLP 不可见的新业务字段；唯一新增信息是 Scenario 已有的 DAG 邻接关系。节点矩阵为 `N_tasks × 4`，合法候选矩阵为 `N_candidates × 14`，候选到节点的索引来自既有动作元组中的 `task_id`。

前驱和后继聚合矩阵均按行归一化。无前驱或无后继节点对应的聚合行保持全零，不添加自环；自身信息由独立 self 分支保留。

## 3. 最小网络

设节点输入为 `X`，一阶前驱/后继均值矩阵为 `A_pred/A_succ`，候选特征为 `C`，候选所属任务为 `t(i)`：

```text
H0 = tanh(X W_node + b_node)
P  = A_pred H0
S  = A_succ H0
HG = tanh(H0 W_self + P W_pred + S W_succ + b_message)
HP_i = tanh(C_i W_pair + HG_t(i) W_context + b_pair)
score_i = HP_i W_output
policy = softmax(score / temperature) over legal candidates only
```

当前固定一次双向消息传递，不把“消息层数”作为调参轴。动作空间仍由环境先产生合法候选，网络只在该集合内 softmax，因此 task-GNN 不具备产生非法 task-resource pair 的输出通道。

## 4. 参数量契约

候选隐藏维为 `h`，消息维为 `g` 时：

```text
task-GNN params = 3g² + 6g + (16 + g)h
MLP params      = 16h
```

最小测试配置 `h=8, g=4` 为 `232` 对 `128` 参数；计划正式配置 `h=32, g=8` 为 `1008` 对 `512` 参数。正式报告必须同时给出绝对参数量、相对倍数、训练时长和 CPU 推理 P50/P95，不得用“模型很小”代替测量。

## 5. checkpoint 与确定性

task-GNN checkpoint 使用无 pickle NPZ，并内嵌：

- `architecture=task_gnn_v1`；
- 14 维基础特征名和 4 维节点特征名；
- `hidden_dim/message_dim/seed`；
- 全部 10 组网络参数。

加载时拒绝 architecture、节点特征、基础特征、参数 shape 或有限性漂移。当前 checkpoint 只覆盖推理参数；Adam、RNG、history、best 选择和事务恢复将在接入 BC/PPO 时沿用 P1-03 状态契约扩展，未完成前不得启动正式 3-seed 训练。

## 6. 第一阶段验收

| 门禁 | 当前证据 | 状态 |
| --- | --- | --- |
| 14 维输入且无 HEFT 决策位 | 构造器只接受规范化 teacher-free schema | 通过自动测试 |
| 合法动作 mask 不变 | cache actions 逐项等于环境 candidates | 通过自动测试 |
| 图结构确实影响表示 | 相同节点/资源、不同 DAG 得到不同 node context | 通过自动测试 |
| 同 seed 确定性 | 参数和 masked probability 逐数组一致 | 通过自动测试 |
| checkpoint 无损往返 | 全参数和 probability 逐数组一致 | 通过自动测试 |
| 参数量可机器读取 | metadata 固定输出维度和 parameter count | 通过自动测试 |
| BC/PPO 梯度与 warm start | 尚未实现 | 未通过，下一提交 |
| PPO epoch 状态与事务恢复 | 尚未扩展到 GNN 参数 | 未通过，下一提交 |
| 固定 3-seed validation 对照 | 尚未运行 | 未开始 |
| B 独立复核 | 尚未进行 | 未开始 |

第一阶段实现位于 `trisched/gnn.py`，聚焦测试位于 `tests/test_task_gnn.py`；本地完整回归为 `176 passed in 7.59s`。本阶段只能表述为“task-GNN 最小表示接口通过本地自动测试”，不能表述为“GNN 已训练”“GNN 优于 MLP”或“P1-A03 已完成”。

## 7. 下一提交要求

1. 为 10 组参数实现解析梯度和 Adam，使用中心有限差分独立核对；
2. 冻结 teacher state 时携带 task graph 张量，使 BC 与 PPO minibatch 不依赖可变环境对象；
3. 用 MLP best/warm-start 的可比规则初始化并记录 task-GNN 来源；
4. 扩展 checkpoint hash、完整 epoch 状态、损坏状态诊断和 staging 恢复；
5. 先运行微型 1-seed 连续/恢复，再申请运行冻结 3-seed validation；
6. 成员 B 从不可变提交复算参数量、CPU 时延、逐实例 win/tie/loss 和失败切片。
