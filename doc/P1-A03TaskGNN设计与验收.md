# P1-A03 task-GNN 设计与验收契约

- 任务：P1-A03 / P1-04
- 主责：成员 A
- 复核：成员 B
- 日期：2026-07-17
- 状态：进行中（B 已独立复核微型 CLI/目录事务；待正式 3-seed）
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

推理 checkpoint 加载时拒绝 architecture、节点特征、基础特征、参数 shape 或有限性漂移。task-GNN PPO 现另有 epoch 边界训练状态，覆盖：

- 当前 actor 10 组参数及两套 Adam 数组、actor RNG；
- 当前 critic 4 组参数及两套 Adam 数组；
- best actor/value 参数、best key 和 best epoch；
- training RNG、完整 history、completed epoch；
- architecture、14 维 schema、hidden/message/value 维度、seed；
- 外部 contract SHA-256、BC warm-start 独立参数 SHA-256、`test_accessed=false`；
- 对 metadata 和 56 个数值数组计算的 payload SHA-256。

状态文件使用同目录临时文件、flush/fsync 和 `os.replace` 原子发布。加载会重算 payload、Adam step 与 history，并拒绝缺失、已存在、损坏、hash、contract、warm-start 或维度错配。CLI 恢复时先把正式目录复制到同盘唯一 staging；配置、teacher/reference、全部 seed、summary 和 manifest 都在 staging 内成功后才交换目录。任一后置校验、训练或写出失败都会删除 staging，正式目录保持逐字节不变。

## 6. 第一阶段验收

| 门禁 | 当前证据 | 状态 |
| --- | --- | --- |
| 14 维输入且无 HEFT 决策位 | 构造器只接受规范化 teacher-free schema | 通过自动测试 |
| 合法动作 mask 不变 | cache actions 逐项等于环境 candidates | 通过自动测试 |
| 图结构确实影响表示 | 相同节点/资源、不同 DAG 得到不同 node context | 通过自动测试 |
| 同 seed 确定性 | 参数和 masked probability 逐数组一致 | 通过自动测试 |
| checkpoint 无损往返 | 全参数和 probability 逐数组一致 | 通过自动测试 |
| 参数量可机器读取 | metadata 固定输出维度和 parameter count | 通过自动测试 |
| 解析梯度与 Adam | log-probability/entropy 对全部 10 组参数做中心有限差分；裁剪前 norm、更新方向和 optimizer clone 已测 | 通过自动测试 |
| frozen graph state | 只读 ranks/归一化上下文/节点/邻接/候选张量可脱离 live env 重放；graph hash 与 Scenario 绑定 | 通过自动测试 |
| BC/PPO 与 warm start | 专用训练函数在 4/2/0 微型数据上完成 BC、PPO、validation 选模和 epoch-0 回退；直接调用仍拒绝 `gamma != 1` | 通过微型 smoke |
| PPO epoch 状态 | 连续/中断/恢复 summary、4 个返回模型和 57 个 NPZ 条目逐数组一致；hash/contract/warm-start/损坏/缺失诊断通过 | 通过微型 smoke |
| 独立配置与 CLI | `stg_task_gnn.json` 固定 architecture/message dim；`train-task-gnn` 不改变 `train-ppo` 默认 MLP | 通过微型 smoke |
| 完整 artifact 与 manifest | 每次 3-seed 微型 run 生成 32 个被 manifest 声明的 artifact；bytes/SHA-256 全部重算一致 | 通过微型 smoke |
| run 级 staging 事务 | 第二个 PPO epoch 中断后恢复与连续 run 一致；后置 seed 写出失败时正式目录逐字节不变且事务残留为 0 | 通过微型 smoke |
| 固定 3-seed validation 对照 | 尚未运行 | 未开始 |
| B 独立复核 | 远端 `f6301ae7...` 干净 clone 全回归、独立 4/2/0 数据、15 对 NPZ、32-artifact、后置失败目录快照和 MLP 旧入口均通过 | [通过](./P1-A03独立复核记录.md) |

实现位于 `trisched/gnn.py`、`trisched/bc.py` 和 `trisched/ppo.py`，聚焦测试位于 `tests/test_task_gnn.py`、`tests/test_task_gnn_training.py` 与 `tests/test_task_gnn_pipeline.py`。第二阶段增加了从候选分数到节点编码的完整解析反传、梯度累加/裁剪 Adam，以及与 Scenario 内容 hash 绑定的只读 `FrozenTaskGraph/FrozenTaskGNNState`。冻结状态在 live env 继续执行后仍逐数组重放相同 probability，错误 Scenario 会在冻结时拒绝。

第三阶段在 `trisched/bc.py` 和 `trisched/ppo.py` 增加专用 task-GNN 训练类型，不修改原 MLP 的 frozen state 或 PPO transition：

- teacher 冻结状态携带只读 graph、合法 actions、14 维候选和 target index，并继续复放 HEFT 后通过双 validator；
- BC 使用相同交叉熵、Adam、validation mean ratio 和 best/last 规则；
- PPO transition 只保存 `FrozenTaskGNNState`、旧 log-probability、只读 critic state、advantage 和 return，不保存 live env；同 episode 共用一个 graph 对象；
- critic、增量 makespan/HEFT 奖励、GAE、clipping、entropy、target-KL 和 validation selection key 与 P1-A02 相同；
- epoch 0 明确作为 task-GNN BC warm start 候选，无改善时必须回退。

微型 smoke 使用合成 `4 train / 2 validation / 0 test`、每图 5–6 tasks、seed 61。BC 两个 epoch 的 validation ratio 为 `2.656002480246 / 2.092007786142`，零失败；PPO epoch 1 仍为 `2.092007786142`，故 best 正确回退 epoch 0。PPO 共 21 个 transition，零失败、零非法动作，最大奖励恒等误差 `8.881784197001252e-16`。新增训练测试 `4 passed`，本地完整回归 `183 passed in 7.65s`。这些数值只证明训练、冻结重放和选模闭环，不是性能收益；ratio 明显差于 1，更不能用作领先证据。

第四阶段使用同一类 4/2/0 合成数据和 seed 79，将 PPO 扩展为 2 epochs，并在第二次 update 前注入中断。恢复状态停在 `completed_epoch=1`；恢复后与连续运行的 summary、best/last actor、best/last critic 和最终状态 57 个 NPZ 条目逐数组一致。状态包含 56 个数值数组，Adam step 由 history 的 update 数与 minibatch 数独立重算；actor/training RNG 均为 PCG64。测试还分别得到 `ppo_resume_state_exists/missing/hash/mismatch/read/corrupt`，其中 mismatch 同时覆盖外部 contract 和 BC warm-start 参数变化，corrupt 额外覆盖重签名后仍多出未知数组的严格 schema 拒绝。新增恢复测试后本地完整回归为 `184 passed in 7.54s`。

第五阶段新增 `configs/stg_task_gnn.json` 和 `train-task-gnn`，把 task-GNN 与现有 `train-ppo` 的 MLP 默认入口分离。微型 CLI 使用 3 seeds、4 train / 2 validation，并物理删除 test raw 文件；每个 seed 输出 BC warm start、actor/value best/last、epoch state、curve、validation diagnostics 和 failure JSONL，run 级另输出 resolved config、双签 teacher/reference、summary 与 manifest。manifest 声明 32 个 artifact，逐个重算 bytes/SHA-256 通过，`test_accessed=false`，首次发布为 `direct_new_directory`。

目录事务测试把 PPO 扩为 2 epochs，在 seed 41 第二次 update 前中断；恢复 run 与连续 run 的 summary、3 条曲线、12 个 actor/value checkpoint 和 3 个训练状态均逐数组一致，manifest 记录 `resumed_seeds=[41]` 与 `publication_mode=staging_directory_swap`。随后在 staging 写 seed 42 曲线时注入异常，正式目录前后逐字节一致，staging/backup 残留为 0。配置测试还拒绝 resource-GNN、课程字段与 14 维 schema 漂移。MLP/task-GNN 聚焦回归通过，完整回归为 `187 passed`。这些仍是事务与闭环证据，不是正式性能证据。

B 在远端不可变提交 `f6301ae7...` 上建立干净 clone，完整回归 `187 passed in 10.33s`；另用 seeds 71/72/73 和物理删除 test 文件的独立 4/2/0 数据复跑。32 个 artifact 全部重算，连续/恢复 15 对 NPZ 逐数组一致，后置 seed 失败保持正式目录逐字节不变且残留为 0；独立 MLP run 仍输出 34 个 artifact。正式 train frozen state 为 6000 transitions/139,848 candidates，唯一张量约 19.74 MiB，采样 RSS 峰值增量约 53.42 MiB，120 个场景恰有 120 个共享 graph。参数量复算为 GNN/MLP `1008/512=1.96875×`。完整方法和带边界的 CPU 预检见 [P1-A03 独立复核记录](./P1-A03独立复核记录.md)。

## 7. 下一步要求

1. 用户推送 B 的独立复核记录后，成员 A 运行冻结 `20260717/20260718/20260719` 正式 validation，公开 test 继续物理隔离；
2. 正式 run 必须使用 `configs/stg_task_gnn.json`，保留全部 32 项 artifact，并记录总训练和冻结前处理时间；
3. 将 task-GNN 与既有 MLP 逐实例配对，报告 win/tie/loss、bootstrap CI、训练时长、正式 checkpoint CPU 推理 P50/P95 和失败切片；
4. 成员 B 从新不可变提交复核正式 summary、checkpoint、manifest、配对统计和 test 禁用边界；
5. 若固定 validation 未产生配对优势或失败/时延明显恶化，保留 MLP 并关闭本轮，不加入 resource-GNN、课程或新奖励补救。
