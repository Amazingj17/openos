# ADR-A01：算法基线与指标口径

- 任务：P0-A01
- 主责：成员 A
- 复核：成员 B
- 状态：已完成（B 复核通过；P1-A03 正式对照与止损已双签）
- 日期：2026-07-16
- 适用版本：首次 GitHub 基线 `28f6305` 之后的 P0-A01 工作区

## 1. 决策结论

1. 主指标固定为逐实例配对比值的算术平均：

   \[
   r_i=\frac{M_{policy,i}}{M_{HEFT,i}},\qquad
   mean\_ratio=\frac{1}{N}\sum_{i=1}^{N}r_i
   \]

   其中 `M` 是通过独立合法性校验后的 makespan。指标越低越好，`< 1` 才表示平均优于 HEFT。

2. 禁止使用“策略平均 makespan / HEFT 平均 makespan”替代逐实例 ratio；禁止删除失败、超时或非法实例。
3. validation 用于选择 checkpoint、奖励、超参数和回退阈值；test 清单冻结后仅用于最终报告，不参与任何模型选择。
4. 开发阶段至少 3 个训练随机种子，最终主结果至少 5 个；最终使用分层 bootstrap 95% CI，并同时公开每个 seed 的结果。
5. 最终主结果必须同时满足：合法调度率 100%、`mean_ratio < 1`、固定 test、逐实例结果可追溯。内部冲奖目标为 `mean_ratio ≤ 0.95`。
6. 当前 masked MLP 的 test `mean_ratio = 1.0`，只能证明训练—评测闭环成立，不能宣称性能领先。
7. 在 G1 正确性门禁通过前不开始 GNN/PPO；正确性通过后，先建立 PPO，再单独验证 task-GNN，禁止同时改变模型、奖励和数据分布。
8. P1-A01 已由 B 在远端不可变提交 `180a713...` 上独立复核通过：完全不含 test 原始字节的 raw root 可连续两次重生成相同核心构件，best/last 复评均为 validation ratio 1.0、零失败、零非法动作。该结论允许开始 P1-A02，但不构成优于 HEFT 的证据。
9. P1-A02 已由 B 在远端不可变提交 `0794d18...` 上独立复核通过：物理删除 test JSON 和相关 archive 后可重生成相同 3-seed validation 指标，checkpoint、31 个 artifact、奖励/GAE/clipping 数学和配置注入均通过。三个 best ratio 均低于 1 且零失败/非法，但 1/3 seed 回退 BC warm start、各 seed P95 均高于 1；最终 test 前仍只能写“validation 门禁通过”。
10. P1-03 在远端不可变提交 `cddca3b...` 上未通过 B 首轮复核：正常连续/恢复轨迹一致，但后置 seed 状态失败会先改写前序 seed artifact，使旧 manifest 失效。A 以 staging 目录交换修复；B 已在 `d27479f...` 上独立重验正常恢复、状态、10 类故障、目录快照和发布回滚并通过，允许开始 task-GNN。
11. P1-A03 正式 3-seed task-GNN mean ratio 为 `0.690460`，对照 MLP 为 `0.723193`；90 对逐实例为 `50/19/21`，但分层配对 bootstrap 95% CI `[-0.083063, 0.007113]` 跨 0。B 已从远端 `f4c7e4d...` 独立重算 artifact、状态、checkpoint 和统计并通过；主模型保留 MLP，task-GNN 仅记为方向性改善、未证实，P1-A03 关闭。

## 2. 本次审计范围与证据

审计文件：

- `trisched/learning.py`：16 维候选特征、masked MLP、HEFT 行为克隆、REINFORCE、checkpoint。
- `trisched/evaluation.py`：逐实例评测、ratio、CI、P50/P95、win/tie、运行时间。
- `trisched/policies.py`：HEFT upward rank、Greedy-EFT、Random。
- `trisched/cli.py`：数据划分、训练、评测和结果输出。
- `configs/smoke.json`：当前 seed、数据规模和训练参数。
- 本地 `outputs/smoke/`：当前 checkpoint、训练历史、逐实例 CSV 和摘要。

本次实际执行：

```powershell
python -m pytest -q
python -m trisched evaluate `
  --config configs/smoke.json `
  --checkpoint outputs/smoke/masked_mlp.npz `
  --split test `
  --output outputs/p0_a01_cli_eval
```

实际结果：`7 passed`；冻结 checkpoint 可在不训练的情况下加载，20 个 test makespan 与原始结果逐实例一致，`mean_ratio = 1.0`。本地 checkpoint SHA-256 为：

```text
7fb644af6ae06530fdac556fd07fc4281a560646d16c2ab5f00d2641fbd8f2b7
```

该 hash 只标识本次本地文件；后续每个正式 run 都必须在自己的 manifest 中记录实际 hash。

## 3. 指标契约

### 3.1 makespan 与合法性

对场景 `i`，makespan 定义为全部任务完成时刻的最大值：

\[
M_i=\max_{v\in V} finish(v)
\]

只有同时满足下列条件的调度才能进入性能统计：

- 每个任务恰好出现一次；
- 所有前驱及跨资源通信在子任务开始前完成；
- 同一资源上的任务区间不重叠；
- 任务执行时长与场景的 workload/speed 一致；
- 报告的 makespan 与任务完成时刻重算值一致。

当前 `run_policy` 会在每个实例结束后依次调用生产 validator 和独立 validator，因此成功结果都经过双检。P0-08 起，统一 evaluator 会保留非 HEFT scheduler 的失败行，按冻结惩罚计入主指标，并从 `success_count/failure_count` 计算真实合法率和失败率；HEFT 基线失败因 ratio 分母不可定义而明确终止。字段和发布规则见[P0-08 失败统计与运行清单](./P0-08失败统计与运行清单.md)。

### 3.2 主指标

每个实例先计算：

```text
policy_ratio_i = policy_makespan_i / heft_makespan_i
```

再对同一冻结 split 的实例等权平均。当前实现遵守这一口径。

解释规则：

| 数值 | 允许结论 |
| --- | --- |
| `< 1.00` 且 CI 上界 `< 1.00` | 有统计证据表明平均优于 HEFT |
| `< 1.00` 但 CI 跨过 `1.00` | 有改善趋势，证据不足 |
| `= 1.00` | 与 HEFT 持平，不得写“优于” |
| `> 1.00` | 平均劣于 HEFT，应触发诊断或回退 |

### 3.3 补充指标

主结果必须同时报告：

- `count`、mean、标准差和 95% CI；
- median/P50、P95；
- win/tie/loss，当前数值容差为 `1e-9`；
- 合法率、失败率和超时率；
- 平均及 P95 推理时延、训练时长、峰值内存；
- 小图相对 CP-SAT/MILP optimum/bound 的 gap；
- 图规模、CCR、资源数量、速度异构和带宽扰动等 OOD 切片。

### 3.4 随机种子与置信区间

当前实现使用总体标准差和正态近似：

```text
mean ± 1.96 × population_std / sqrt(N)
```

这在样本少、ratio 偏态或长尾时不够稳健。正式报告改为：

1. 每个训练 seed 在同一批 test 实例上得到配对 ratio。
2. 至少 5 个训练 seed，公开全部 seed-level mean。
3. 使用交叉分层 bootstrap：有放回抽取 seed，并有放回抽取所有 seed 共享的场景 ID，重复至少 10,000 次。
4. 报告 bootstrap 2.5%/97.5% 分位数；正态近似只保留作调试信息。
5. 所有 slice 同时给出样本数，样本过少时不得单独作强结论。

P1-B02 将上述方法固化为机器契约：主策略和 Random 使用 5 seeds；确定性 baseline、单 seed BC 和既有 3-seed task-GNN 不复制行伪造相同 seed 数。主策略与 seed 数不同的 comparator 比较时，双方 seed 独立重采样、场景索引共享配对；只有 seed 集完全一致时才额外报告逐 seed×scenario 精确配对。A 首轮因报告没有输出全部 seed-level mean 而退回；B 在 JSON/逐 seed CSV/manifest 补齐后，A 已用 mean `0.6/1.4` 的不同 seed 独立[二次复核通过](./P1-B02第二轮独立复核记录.md)。

### 3.5 失败、非法和超时

- 非法或不完整调度触发一票否决：该 run 不具备主结果发布资格。
- 评测器仍需保留失败实例行，写明异常类型和 `failure_rate`，禁止因异常删除困难实例。
- 超时阈值和惩罚规则必须在 validation 上冻结；test 开始后不得调整。
- 带失败的 run 即使已生成完整统计也不具备主结果发布资格；CLI 必须非零退出，报告必须同时给出惩罚后主指标和 failure rate。

## 4. 当前实现审计

| 项目 | 当前实现 | 审计结论 | 后续动作 |
| --- | --- | --- | --- |
| 逐实例 ratio | 每个策略 makespan 除以同实例 HEFT makespan | 正确 | 保持 |
| train/validation/test | smoke 使用派生 seed；公开 BC/PPO 只用 train 生成 teacher/梯度，validation 选模，test 不加载 | P1-A01/P1-A02 均已由 B 用无 test raw root 复核；PPO 复跑时 test JSON 和 archive 均物理不存在 | test 只留最终评测；task-GNN 继续复用同一门禁 |
| checkpoint 选择 | legacy smoke 只存 last；公开 BC/PPO 均保存 best/last，PPO 将 warm start 作为 epoch 0 候选 | B 证明 seed 20260718 的 best actor 参数与 BC warm start 相同，未用 last 冒充 best | task-GNN 沿用相同 selection key |
| CI | smoke 仍为正态近似；P1-A03 对照新增固定双层配对 bootstrap；P1-B02 已冻结 5-seed 与异 seed 数 comparator 统计 | OOD source/hash/外层计时通过 A 独立复核，但 runner 内部非法漏计和错误返回类型被退回 | B 修复失败语义、A 二次复核后，再进入 5-seed development/OOD |
| 合法率 | 从逐实例成功/失败计数计算，失败进入 CSV/JSONL | P0-08 已实现并由 A 独立复核 | 保持零失败发布门禁 |
| HEFT teacher | P1-A01 参考含 `is_heft_task/is_heft_pair`；P1-A02 主路径强制删除 | 同 seed 16/14 维 BC ratio 为 `1.0/0.852539`，teacher accuracy 为 `1.0/0.514`；B 注入任一特征均在训练前失败 | PPO/GNN 主路径保持删除 |
| 动作空间 | 对 `ready task × resource` 联合候选打分 | 有严格 mask，但规模为乘积 | PPO 阶段比较两阶段因子化策略 |
| 奖励 | legacy 为终局 `-ratio`；P1-A02 为逐步 `-(C_t-C_{t-1})/M_HEFT`，和严格等于 `-ratio` | B 独立推导通过；正式最大恒等式误差 `1.78e-15`，`gamma=0.99` 注入被拒绝 | task-GNN 保持奖励与 `gamma=1.0` 不变 |
| 训练算法 | legacy episodic REINFORCE；公开路径为 BC warm start + clipped PPO/GAE/value/target-KL | B 已独立复核 3-seed MLP、staging 断点续训和 task-GNN 同预算正式对照 | 保持 MLP 主路径，转入 P1-B02 评测契约 |
| 图表示 | PPO 主模型为 14 维 MLP；P1-A03 对照为一层双向 task-GNN | GNN mean 点估计改善 `-0.032733`，但配对 CI 跨 0；参数约 `1.969×`、CPU P50 约 `1.046×` | 保留 MLP；GNN 作为未证实消融，不追加新图变量 |
| checkpoint 元数据 | checkpoint 自带维度/seed/特征；run manifest 记录配置、数据、代码、依赖和 checkpoint hash | P0-08 已实现外部清单；P1-A03 合同可由历史 Git blob 重建，但 clean clone 的 LF/CRLF 工作树字节不同 | release 前规范化文本 hash、补跨 clone 恢复测试，再考虑内嵌 manifest 摘要 |
| test 使用 | legacy `pipeline` 每次评测合成 test；公开 `train-bc/train-ppo` 对 test 设用途门禁；P1-B02 development materializer 固定只加载 `validation/model_selection` | A 二次确认 claim/final basename、receipt 字段和 `utc_seconds_z`；B 的 OOD 候选记录 `test_accessed=false`，public-test loader 仍未接入 | A 复核无 test 边界；最终将 claim 固定在 public-test loader 前并固定 release root |

## 5. 当前结果的正确解释

| 项目 | 当前值 | 解释 |
| --- | --- | --- |
| 数据 | train 24 / validation 10 / test 20 | 只能做工程 smoke，不能支持比赛结论 |
| 图规模 | 8–16 tasks，3 resources | 没有大图和资源数泛化证据 |
| BC epoch 3 action accuracy | 1.0 | 模型已能复制 teacher；不等于优于 teacher |
| REINFORCE epoch 5 训练 mean ratio | 约 0.99923 | 来自随机采样训练轨迹，不能替代冻结策略测试结果 |
| deterministic test mean ratio | 1.0 | 与 HEFT 完全持平 |
| test win/tie/loss | 0% / 100% / 0% | 20 个实例全部复制 HEFT 结果 |
| test 合法率 | 100%，failure rate 0% | P0-08 的逐实例真实计数；仍只覆盖当前 smoke |
| 平均推理时间 | MLP 约 3.003 ms；HEFT 约 1.162 ms | 当前 MLP 约为 HEFT 的 2.58 倍，未体现工程效率优势 |

因此当前唯一允许的表述是：

> TriSched MVP 已打通受约束策略训练、checkpoint 保存/加载和统一评测闭环；在当前 20 个合成测试实例上，masked MLP 复现 HEFT 调度，尚未获得性能提升。

P1-A01 另在公开 STG topology projection 上冻结了 120 个 train teacher 和 30 个 validation reference。单 seed 纯 BC 在 epoch 1 达到 validation teacher action accuracy 1.0、mean ratio 1.0、failure/illegal action rate 0；test 未访问。B 已用完全不含 test 字节的数据根双次重跑并独立复核通过。该结果同样只允许表述为“公开数据上复现 HEFT”，详见 [P1-A01 记录](./P1-A01HEFT教师与BC基线.md)及其[独立复核记录](./P1-A01独立复核记录.md)。

P1-A02 删除两项直接 HEFT 决策特征后，在相同公开 train/validation 契约上完成 BC warm start + masked PPO。3 个 best validation ratio 为 `0.807240/0.623254/0.739086`，mean/std 为 `0.723193/0.075948`，failure/illegal 均为 0；PPO 改善 2 个 seed，另 1 个回退 warm start。B 已在完全无 test 字节的数据根上复跑并独立复核通过。由于每个 seed 仍有退化实例且 P95 ratio 均大于 1，当前仍只允许表述为“3-seed validation 开发门禁通过，尚无稳定 PPO 收益或最终泛化结论”，详见 [P1-A02 设计与验收](./P1-A02MaskedPPO设计与验收.md)及其[独立复核记录](./P1-A02独立复核记录.md)。

P1-A03 在相同 14 维输入、奖励、split、训练预算、3 seeds 和选模规则下，只把 MLP 替换为一层双向 task-GNN。三个 best validation ratio 为 `0.754139/0.633578/0.683664`，mean/std 为 `0.690460/0.049453`，failure/illegal 均为 0；与 MLP 的 90 对逐实例为 `50/19/21`。分层配对 bootstrap 95% CI 为 `[-0.083063, 0.007113]`，仍跨 0，因此当前只允许表述为“task-GNN 在 validation 上呈方向性改善，但未达到稳健替换门禁”。B 已独立复核通过，主模型继续保留 MLP，公开 test/OOD 未访问；详见 [P1-A03 正式对照报告](./P1-A03正式对照报告.md)和[正式结果独立复核](./P1-A03正式结果独立复核记录.md)。

P1-B02 已把下一阶段的主策略、reference、policy seed 数、ID/OOD 切片、失败惩罚、scheduler-only runtime、交叉分层 bootstrap、主策略配对比较、artifact hash 和一次性 test 工作流写入机器契约。契约/聚合器子阶段已通过；30 个 ID 与三类各 30 个 OOD 场景的完整 payload、source/hash、无 test 路径和外层计时也由 A 独立重算通过。但 evidence producer 会把 runner 内部已判非法的记录写成零 illegal，并让错误返回类型泄漏 `AttributeError`，因此 A [退回 OOD 证据路径](./P1-B02OOD证据路径独立复核记录.md)。5-seed 主模型尚未补齐，也没有任何 OOD 性能结果，不能把“场景已生成”写成“泛化已验证”。

## 6. 当前 MLP/PPO 与 legacy REINFORCE 的局限

1. **teacher 泄漏边界**：P1-A01 的 16 维 BC 参考仍直接包含 HEFT 决策；P1-A02 主路径已强制删除，并由 B 完成训练前注入复核。
2. **无图消息传递**：入度、出度和 upward rank 不能完整表达多跳依赖、关键汇合点和并行分支竞争。
3. **资源关系被压扁**：资源只以单候选的速度、类型和当前可用时刻出现，没有资源—链路图的联合表示。
4. **联合候选扩张**：动作数为 ready tasks 与 resources 的乘积，大图上采样、归一化和探索成本都会上升。
5. **奖励 credit 仍粗糙**：P1-A02 的增量 makespan reward 与终局 ratio 严格等价，但一个早期选择的通信后果可能延迟到后续步骤才体现。
6. **PPO 收益不稳定**：虽然已有 value、GAE、clipping 和 KL 诊断，仍有 1/3 seed 回退 warm start，seed std 为 `0.07595`。
7. **训练与部署策略不一致**：PPO 训练随机采样、validation 使用确定性 argmax；训练均值不能替代部署策略结果。
8. **两条训练路径并存**：公开 PPO 已统一 best/last 与 warm-start 回退，legacy smoke 仍只保存最终 REINFORCE 模型。
9. **公开数据仍只有 validation 证据**：P1-A02 已完成独立复核且不再只是复制 HEFT，但尚无公开 test、CCR、OOD 或外部泛化证据。
10. **统计证据不足**：已有 3 个开发 seed × 30 validation 和分层配对 bootstrap，但 CI 跨 0；尚无最终 5-seed 或一次性 test/OOD 结果。
11. **失败惩罚仍需 benchmark 冻结**：P0-08 已实现真实失败记录和可配置惩罚，但默认 10.0 只是工程值，正式实验必须只在 validation 上冻结。
12. **checkpoint 未内嵌完整元数据**：外部 run manifest 已记录配置、数据、commit、依赖和 checkpoint hash；模型文件本身仍只保存维度、seed、特征名和权重。

### 算法推进顺序

G1 通过后的唯一允许顺序：

1. 去掉/保留 HEFT 二值特征做严格消融，建立 BC-only 基线。（已由 B 复核）
2. 在相同 MLP 和相同数据上用 PPO + GAE 替换 REINFORCE。（3 seeds 已由 B 复核）
3. P1-03 断点续训首轮被 B 退回，A 实现 staging 发布后已通过 B 的第二轮独立复核。（已完成）
4. 只把 MLP 替换为 task-GNN 的正式对照和 B 独立复核均已完成；点估计改善但 CI 跨 0，按止损保留 MLP，不以追加 resource-GNN、课程或新奖励补救。（已关闭）

## 7. 三个独立手算案例

以下案例先按公式手算，再用现有 HEFT 和 validator 交叉验证。它们是 P0-A02 十个黄金案例的首批候选，尚需 B 将其固化为独立 fixture。

### 案例 1：链式 DAG，同资源避免通信

定义：

- 任务：`T0(workload=4) → T1(workload=2)`，边数据量为 2。
- 资源：`R0(device, speed=1)`，`R1(cloud, speed=2)`。
- 跨资源带宽为 2、时延为 0，因此跨资源通信时间为 `2/2=1`；同资源为 0。

执行时间：

| 任务 | R0 | R1 |
| --- | ---: | ---: |
| T0 | 4 | 2 |
| T1 | 2 | 1 |

upward rank：

```text
rank(T1) = mean(2, 1) = 1.5
rank(T0) = mean(4, 2) + avg_comm(1) + rank(T1) = 5.5
```

HEFT 手算：

| 顺序 | 任务 | 资源 | 开始 | 完成 | 选择理由 |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | T0 | R1 | 0 | 2 | R1 完成时刻 2 小于 R0 的 4 |
| 2 | T1 | R1 | 2 | 3 | 同资源无通信；R0 需在 3 开始、5 完成 |

结果：`HEFT makespan = 3.0`。该案例检查链式依赖、执行速度和同资源零通信。

### 案例 2：fork DAG，并行资源与 tie-break

定义：

- 任务：`T0(workload=2)`，`T1(workload=4)`，`T2(workload=4)`。
- 依赖：`T0 → T1`、`T0 → T2`，两条边数据量为 0。
- 资源：`R0(device, speed=1)`，`R1(edge, speed=2)`；通信为 0。

upward rank：

```text
rank(T1) = rank(T2) = mean(4, 2) = 3.0
rank(T0) = mean(2, 1) + 3.0 = 4.5
```

HEFT 手算：

| 顺序 | 任务 | 资源 | 开始 | 完成 | 选择理由 |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | T0 | R1 | 0 | 1 | 最早完成 |
| 2 | T1 | R1 | 1 | 3 | T1/T2 rank 相同，先选 ID 小的 T1 |
| 3 | T2 | R0 | 1 | 5 | R0 与 R1 都在 5 完成，资源 ID 小的 R0 胜出 |

结果：`HEFT makespan = 5.0`。该案例检查 fork 就绪集、并行调度和确定性 tie-break。

### 案例 3：菱形 DAG，HEFT 局部最优造成全局损失

定义：

- 任务 workload：`T0=4`、`T1=4`、`T2=8`、`T3=2`。
- 依赖及数据：`T0→T1:2`、`T0→T2:4`、`T1→T3:2`、`T2→T3:2`。
- `R0=edge, speed=2`；`R1=cloud, speed=4`。
- `R0→R1`：带宽 2、时延 0.5；`R1→R0`：带宽 4、时延 0.25。

执行时间：

| 任务 | R0 | R1 | 平均 |
| --- | ---: | ---: | ---: |
| T0 | 2 | 1 | 1.5 |
| T1 | 2 | 1 | 1.5 |
| T2 | 4 | 2 | 3.0 |
| T3 | 1 | 0.5 | 0.75 |

对数据量 2，两个跨资源方向的通信时间分别为 `1.5` 和 `0.75`，平均 `1.125`；对数据量 4，分别为 `2.5` 和 `1.25`，平均 `1.875`。

upward rank：

```text
rank(T3) = 0.75
rank(T1) = 1.5 + 1.125 + 0.75 = 3.375
rank(T2) = 3.0 + 1.125 + 0.75 = 4.875
rank(T0) = 1.5 + max(1.125 + 3.375, 1.875 + 4.875) = 8.25
```

HEFT 手算：

| 顺序 | 任务 | 资源 | 开始 | 完成 | 关键计算 |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | T0 | R1 | 0 | 1 | cloud 最快 |
| 2 | T2 | R1 | 1 | 3 | T2 rank 高于 T1 |
| 3 | T1 | R0 | 1.75 | 3.75 | `T0` 从 R1 到 R0 通信 0.75；比在 R1 完成于 4 更早 |
| 4 | T3 | R0 | 3.75 | 4.75 | 等待 T1=3.75，及 T2 从 R1 到 R0：`3+0.75=3.75` |

结果：`HEFT makespan = 4.75`。

合法替代方案把四个任务都放到 R1：

```text
T0: 0–1, T2: 1–3, T1: 3–4, T3: 4–4.5
```

其 makespan 为 `4.50`，相对 HEFT：

\[
ratio=4.50/4.75=18/19\approx0.94737
\]

这个案例证明 HEFT 对 T1 选择“局部更早完成”的 R0，会给汇合任务 T3 引入通信和资源约束；它是学习策略应捕捉的全局结构信号，也是 task-GNN 可能产生价值的最小反例。

## 8. 复核状态与下一冻结契约

P0-A01 的指标解释、checkpoint 加载和三个手算案例已由 B 独立复核通过。P1-A02 又在同一指标契约上完成更严格的无 test 正式复跑：远端 7 环境全绿，31 个 artifact、180 份 validation checkpoint 复评、奖励/GAE/clipping 数学与配置注入全部通过，完整证据见 [P1-A02 独立复核记录](./P1-A02独立复核记录.md)。

P1-03 epoch 边界状态覆盖 actor/value 参数、两个 Adam、两套 RNG、history 与 best 选择，并绑定配置、代码、数据、teacher/reference 和 warm start；B 在首轮复核中发现 seed 32 状态失败前 seed 31 artifact 被重写，A 改为 staging 内完成全部恢复、成功后目录交换。B 已在第二轮从新远端提交独立验证正常恢复、10 类故障、目录快照和发布回滚并通过；完整过程见[首轮复核](./P1-03独立复核记录.md)、[第二轮复核](./P1-03第二轮独立复核记录.md)及[设计与验收](./P1-03PPO断点续训设计与验收.md)。P1-A03 随后按以下冻结项执行：

- 沿用相同 14 维基础输入及特征含义，不恢复 HEFT 决策位；
- 沿用增量 makespan/HEFT 奖励、`gamma=1.0`、相同 split 和 test 禁用门禁；
- 沿用 `20260717/20260718/20260719`、checkpoint selection key 和 warm-start 回退；
- 第一轮只将 MLP 表示替换为 task-GNN，不同时加入 resource-GNN、课程、OOD 或新奖励；
- 正式 task-GNN 训练前，P1-03 断点续训及中断/损坏状态注入必须由 B 复核通过（已满足）；
- 报告逐实例配对结果、参数量、CPU 推理 P50/P95 和失败切片，未出现固定 validation 配对优势则回退 MLP。

P1-A03 已按上述冻结项实现并运行：节点只复用 14 维输入中的 workload、upward-rank、indegree、outdegree，经一次前驱/后继均值消息传递后与候选表示融合。B 已从远端 `f6301ae7...` 独立重验微型事务；A 随后在提交 `a8c08c0...` 上完成正式 3-seed；B 又从远端 `f4c7e4d...` 重算 31/32-artifact、三个状态、配对 CI，并用六个 checkpoint 复评 180 次 validation 调度，全部一致。GNN/MLP mean 为 `0.690460/0.723193`，90 对为 `50/19/21`，但配对 CI 跨 0；因此停止扩展并保留 MLP。完整证据见 [P1-A03 设计与验收](./P1-A03TaskGNN设计与验收.md)、[微型独立复核](./P1-A03独立复核记录.md)、[正式对照报告](./P1-A03正式对照报告.md)和[正式结果独立复核](./P1-A03正式结果独立复核记录.md)。

P1-B02 首轮历史见 [独立复核记录](./P1-B02独立复核记录.md)；B 修复后的逐 seed mean、receipt basename 和 UTC 已由 A 从 `071b861...` [二次复核通过](./P1-B02第二轮独立复核记录.md)，契约/聚合器子阶段关闭。OOD materializer/manifest 已通过 A 的独立重算，producer 外层计时也正确；当前门禁改为 B 修复“内部非法漏计”和“错误返回类型未结构化”，再由 A 从新不可变提交二次复核。本阶段仍不读取公开 test，不启动新模型，也不运行 5-seed development/OOD。
