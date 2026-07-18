# P1-A05 Size-OOD 稳健性设计与预注册

- 任务：`P1-A05-SIZE-ROBUSTNESS-DESIGN`
- 主责：成员 A
- 复核：成员 B
- 日期：2026-07-18
- 状态：设计候选已完成，等待 B 从不可变提交独立复核；尚未训练
- 前置证据：P1-B02 development/OOD 已由 A 独立复跑通过，但性能门禁失败
- 边界：只分析 `id_validation` 与 `ood_size`；未加载 public test、未运行 gate、未启动训练

## 1. 决策结论

P1-A05 只预注册一个候选：**在保持每 seed、每 epoch 恰好 6,000 个 PPO transition 不变的前提下，把原来全由 50-task STG 构成的 rollout 改成 3,000 个 50-task STG transition + 3,000 个独立 100-task synthetic transition。**

本候选只改变 PPO rollout 的场景来源计划。以下内容全部冻结不变：

- 14 维 masked MLP、隐藏层宽度和五个训练 seed；
- P1-A04 每 seed 的 BC warm-start actor 字节；
- 奖励、`gamma`、GAE、PPO clip、熵、KL、学习率、minibatch 和梯度裁剪；
- 两个 PPO epoch 和每 epoch 6,000 transition 的预算；
- 只用冻结 ID validation 选择 warm start/epoch 1/epoch 2 checkpoint；
- P1-B02 的四个 development 切片、失败语义、统计与发布门禁。

机器可读预注册为 [`configs/p1_a05_size_robustness_preregister.json`](../configs/p1_a05_size_robustness_preregister.json)，当前 canonical JSON SHA-256 为：

```text
9ab60842d87e6d906ab18ad135cda2a372be569ad39b28ce6f17c60f1ff64647
```

B 签字前不得实现正式训练入口或启动优化。即使候选形式上合理，也不能把本设计状态写成“size 泛化已修复”。

## 2. 冻结证据与可复现命令

根因分析入口：

```powershell
python scripts/analyze_p1_a05_size_robustness.py
```

该命令只读已经双签的 development materialization/evidence 和 P1-A04 训练摘要，沿 HEFT 轨迹统计场景、候选数和 5 个固定进度点的 14 维候选特征，不装载任何 public-test 文件，也不训练或评测新策略。

| 证据 | SHA-256 |
| --- | --- |
| 根因分析脚本 LF 规范化 | `9ca6f0f029d3e947c2be5a17dda1627273c29db72713d753e5eef523b4e354cf` |
| 忽略目录根因报告 | `987b312d73e0e35e124b34bd917f29aa4a97b762e6bdb1c2c0ae2f07c245f41d` |
| P1-A04 五 seed 训练配置 canonical JSON | `fb4a8dd6f304a24c45ce5ad014e7d489aa22d041a7bdb74fadf611c4439f5d04` |
| P1-B02 evaluation contract canonical JSON | `a46986f5695d59612ad9bf2c3379393620418b3f324a1003b5b5772b110c639e` |
| B 正式 development evidence 文件 | `40d781cad7dff39553ba0c027b686e5710d19ceed62fa3bf7f0471a133feb744` |
| evidence `records` | `0cf814874381d2ad392b3ae193fe11726949eefed72500a1f45433953fa8f8dd` |
| P1-A04 五 seed training summary | `3d06ecd6e211fb5a5731c9138ba15cd672ad18b06f3fadb9d1ad4bc6420e3178` |

报告边界字段为：

```text
test_accessed=false
public_test_loaded=false
training_started=false
analyzed_slices=id_validation,ood_size
```

提交前验证为：

```text
P1-A05 preregistration tests = 4 passed
full regression = 236 passed in 12.06s
Black check = passed
compileall = passed
JSON/Markdown link/fence/whitespace checks = passed
```

根因报告位于 Git 忽略目录：

```text
outputs/p1-a05-size-robustness-design/root_cause_analysis.json
```

不把本机报告伪装成 Git 源文件。B 复核时应从不可变提交和已接受的输入 artifact 独立重生成，并核对脚本 hash、输入 hash 和报告 hash。跟踪 JSON 使用 canonical hash，脚本使用 LF 规范化 hash，报告不嵌入 commit 或 wall-clock 字段，因此相同语义输入与脚本应跨提交、跨 LF/CRLF worktree 生成相同字节。

## 3. 根因证据

### 3.1 失败不是单 seed 或少数困难图

| 切片/策略 | 记录数 | mean ratio | 最小值 | 最大值 | win/tie/loss |
| --- | ---: | ---: | ---: | ---: | --- |
| ID / masked MLP | 150 | 0.709190 | 0.136299 | 1.477630 | 124/0/26 |
| ID / task-GNN | 90 | 0.690460 | 0.135665 | 1.413660 | 78/0/12 |
| size / masked MLP | 150 | 1.568302 | 1.230884 | 1.744119 | 0/0/150 |
| size / task-GNN | 90 | 1.573522 | 1.275675 | 1.744119 | 0/0/90 |
| size / 16-D HEFT-flag BC | 30 | 1.000000 | 1.000000 | 1.000000 | 0/30/0 |

结论：五个 MLP seed 的 150 条 size 记录和三个 task-GNN seed 的 90 条 size 记录全部劣于 HEFT。退化覆盖所有 seed 和所有冻结场景，不是由一个坏 seed 或几个长尾图驱动。

16-D BC 之所以恒等于 HEFT，是因为其冻结 schema 显式含 `is_heft_task/is_heft_pair`，只能证明调度环境和 teacher 路径在 100-task 图上仍工作，不能作为无 teacher 特征主模型的泛化证据。

### 3.2 失败不只是 PPO 把 warm start 推坏

P1-A04 seed `20260718` 的正式 best checkpoint 按 ID 选模规则直接保留 epoch 0 BC warm start，没有选择任何 PPO epoch。它在 size-OOD 上的 mean ratio 仍为：

```text
1.462685574891456
```

因此“只增加 PPO-to-BC KL”或“减小 PPO 更新”不能由现有证据单独支持。PPO 更新可能影响具体数值，但不是 size 失败的必要条件。

### 3.3 task-GNN 没有解决该退化

task-GNN 的 size mean ratio 为 `1.573522`，且 90/90 条均为 loss，与 MLP 的失败方向一致。这说明“一层 task-GNN 替换 MLP”已经被现有正式对照否定为充分修复，不应重新启动同一个已关闭假设，也不应无预注册地追加 resource-GNN。

### 3.4 `ood_size` 是复合偏移，不是纯 size 实验

| 描述量 | ID validation | size-OOD | 解释 |
| --- | ---: | ---: | --- |
| tasks | 固定 50 | 固定 100 | 轨迹长度和决策机会翻倍 |
| progress 步长 | 0.02 | 0.01 | 同一网络看到更细的进度状态 |
| edges mean | 332.17 | 684.13 | 绝对依赖数增加 |
| edge density mean | 0.2712 | 0.1382 | 图密度分布同时变化 |
| roots mean | 11.27 | 1.00 | synthetic 生成器强制形成单根结构 |
| longest path mean | 9.60 | 26.03 | 依赖深度约增至 2.71 倍 |
| workload mean | 10.81 | 7.10 | 工作量分布不同 |
| max workload mean | 32.00 | 11.89 | ID 长尾任务在 synthetic 中不存在 |
| off-diagonal bandwidth min | 100.0 | mean 3.36 | 通信系统不在同一范围 |
| off-diagonal bandwidth max | 10000.0 | mean 12.76 | edge-cloud 链路量级完全不同 |
| off-diagonal latency mean | 0.0 | 0.3711 | synthetic 额外引入非零时延 |

ID 来自 STG rnc50 projection；size-OOD 来自 `generate_scenario(task_count=100, edge_probability=0.12)`。除任务数外，拓扑族、工作量、资源速度扰动、带宽和时延一起变化，所以当前只能诊断“固定 50-task STG 训练对 100-task synthetic 域不稳健”，不能作纯任务规模的因果声明。

### 3.5 特征归一化避免了数值爆炸，但没有消除结构偏移

14 维候选特征按每个场景的 workload、speed、time、rank 和 degree 归一化。固定进度点检查显示，大部分 size 候选特征仍落在 ID 的 min/max 范围内；最大的均值偏移约为：

- `task_workload`：`+0.670` 个 ID 标准差；
- `progress`：`+0.482` 个 ID 标准差；
- `communication_delay`：`-0.475` 个 ID 标准差；
- `earliest_finish`：`-0.460` 个 ID 标准差；
- `execution_time`：`+0.460` 个 ID 标准差。

只有 `resource_speed` 有约 `19.7%` 的 size 候选超出 ID 原始 min/max，原因是 ID 资源速度固定而 synthetic 资源速度含 ±15% 扰动。这排除了简单的“某个连续特征整体数值爆炸”解释，但不排除更长轨迹、单根深图和通信域共同造成的决策误差累积。

## 4. 选择与拒绝的候选

| 候选 | 决定 | 原因 |
| --- | --- | --- |
| 重新启用 task-GNN/resource-GNN | 拒绝 | task-GNN 已在 90/90 size 记录上失败；追加 resource-GNN 会同时改架构和训练分布 |
| 只加强 PPO 的 BC/KL 约束 | 拒绝 | 未经 PPO 更新的正式 warm-start seed 仍明显失败 |
| size>50 时直接回退 HEFT | 拒绝为主候选 | 可把 ratio 限制到 1.0，但无法满足严格 `<1` 的全切片发布门禁 |
| 用当前 30 个 size 场景选 checkpoint | 禁止 | 会把冻结 development-OOD 变成反复调参集 |
| 固定预算的 50/100-task rollout 混合 | 采用 | 直接作用于现有证据支持的训练分布/轨迹错配，同时保持模型、目标和 transition 预算不变 |

## 5. 单一候选的冻结实现

### 5.1 数据计划

BC warm start 继续逐字节复用 P1-A04 的五个正式 actor，不重训。只改变 PPO rollout：

| PPO epoch | STG train | synthetic train | 总 episode | 总 transition |
| --- | --- | --- | ---: | ---: |
| 1 | split index 0–59，60×50 = 3000 | seed `20261001..20261030`，30×100 = 3000 | 90 | 6000 |
| 2 | split index 60–119，60×50 = 3000 | seed `20261031..20261060`，30×100 = 3000 | 90 | 6000 |

synthetic 配置固定为：

```text
generator=trisched.scenario.generate_scenario
task_count=100
resource_count=3
edge_probability=0.12
scenario_id=p1-a05-train-size-{index:04d}
```

60 个场景必须在任何策略运行前一次性物化并冻结 manifest。它们与 P1-B02 size-OOD 的 `20260801..20260830` 不重合；仍须用 content hash 对 benchmark train/validation 和四个 development 切片做零重复检查，不能只相信 seed 不同。

### 5.2 唯一允许变化的接口

实现只允许新增一个显式 rollout source plan，使每个 epoch 按冻结清单装载 60 个 STG 和 30 个 synthetic 场景。由于任务长度不同，`episodes_per_epoch` 从 120 联动变为 90，但 transition 数严格保持 6,000；这是同一 sampler 干预的预算约束，不是第二个训练假设。

任何 reward、特征、网络、优化器、epoch、seed、checkpoint selection 或 evaluation 变化都应在配置校验阶段被拒绝。实现不得隐式读取文件夹中“所有 JSON”，必须只读 manifest 声明的文件。

### 5.3 选模与正式评测

每 seed 仍只在 30 个冻结 ID validation 场景上比较：

```text
BC warm start / PPO epoch 1 / PPO epoch 2
→ zero failures
→ lower ID validation mean_ratio
→ earlier epoch
```

`ood_size` 不参与 checkpoint selection。五个 checkpoint 全部冻结后，才允许用未改变的 P1-B02 development 契约正式运行一次四切片报告。

## 6. 预算、成功标准与停止规则

### 6.1 正式预算

```text
5 seeds × 2 PPO epochs × 6000 transitions = 60000 transitions
formal training run count = 1
formal development evaluation count = 1
public test access count = 0
```

基础 P1-A04 同样是每 seed 每 epoch 6,000 transition，因此比较保持总决策预算一致。允许按 P1-03 合同从精确事务状态恢复基础设施中断，不允许把恢复变成更换 seed 或重跑挑结果。

### 6.2 必须全部满足的成功标准

1. 自动报告的三个原有 gate 必须同时为真：

   ```text
   primary_zero_failures_and_illegal_actions=true
   primary_mean_ratio_below_reference_on_every_reported_slice=true
   release_publishable=true
   ```

2. 新模型 size-OOD mean ratio `<1.0`，且 new−P1-A04 的逐 seed×scenario 配对 bootstrap 95% CI 上界 `<0`。
3. 新模型 ID mean ratio `<1.0`，并且不高于已接受值 `0.709190 + 0.02 = 0.729190`。
4. CCR-OOD 和 system-OOD mean ratio 均 `<1.0`；P1-B02 原合同仍是发布结论的权威来源。
5. 零 failure、timeout、penalty 和 illegal action；全部 artifact/checkpoint/config/source hash 可重算。

### 6.3 停止规则

- B 未通过设计或任一训练前门禁失败：不训练，只修复设计/实现缺陷后重新送审。
- 唯一正式运行完成但任一成功标准失败：保留 P1-A04 masked MLP，记录负结果并关闭本候选，不调 sampler 比例、seed、epoch、学习率或选模规则。
- 若要尝试其他数据比例、课程、KL、架构、reward 或回退，必须创建新 ADR 和新任务 ID，不能在 P1-A05 下补跑。
- P1-A05 即使 development 通过，也须由 B 从不可变提交独立复跑；B 签字前不启动 G3。

## 7. B 的独立复核清单

B 应从用户推送后的新远端提交建立干净 detached worktree，并至少完成：

1. 重跑根因分析，核对输入/脚本/报告 hash 和三个禁止字段；
2. 手算固定训练 horizon `6000/120=50`，核对 size horizon 恰为 100；
3. 从原始 evidence 重算 MLP `0/0/150`、task-GNN `0/0/90` 和 warm-start seed `1.462686`；
4. 抽查 ID/size 场景描述量，确认 `ood_size` 是复合偏移而非纯 size；
5. 重算预注册 JSON hash，检查每 epoch `3000+3000=6000` 和总预算 60,000；
6. 注入重叠 seed/hash、91 episodes、非 6,000 transitions、额外配置变化、public-test 路径与训练调用，确认训练前全部拒绝；
7. 确认本提交没有新 checkpoint、训练曲线、优化器状态或 public-test receipt。

B 只有在上述设计和机器约束均通过后，才可签字允许 A 进入后续 `P1-A05-IMPLEMENT`。本记录本身不授权训练。
