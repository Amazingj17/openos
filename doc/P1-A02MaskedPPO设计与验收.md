# P1-A02 Masked PPO 设计与验收契约

- 任务：P1-A02 / P1-03
- 主责：成员 A
- 复核：成员 B
- 日期：2026-07-17
- 状态：部分完成（实现与测试已提交，待干净提交正式运行和 B 独立复核）
- 前置门禁：P1-A01 / P1-02 已完成（B 独立复核通过）
- 数据契约：`stg-rnc50-hetero-trisched-v1`

## 1. 目标与非目标

本任务在 P1-A01 的公开 STG train/validation 数据契约上，用 clipped PPO + GAE 替换 legacy smoke 中的 episodic REINFORCE。策略继续只从合法的 `ready task × resource` 候选中采样，训练、checkpoint 选择和止损判断均不得访问 test。

本任务只验证相同 MLP 表示下的优化算法变化，不同时加入 task-GNN、resource-GNN、动态系统或 OOD 课程。PPO 未在至少 3 个 seed 上通过 validation 止损线时，不开始 P1-A03 task-GNN，也不得宣称优于 HEFT。

## 2. 受控特征消融

P1-A01 的 16 维候选特征包含：

```text
is_heft_task
is_heft_pair
```

两者直接暴露 HEFT 当前动作，只适合作为 BC 管线 sanity reference。P1-A02 的主 BC warm start 和全部 PPO seed 强制删除这两维，只保留其余 14 维。配置若没有同时排除这两项，必须以稳定错误码 `ppo_teacher_feature_leakage` 失败。

正式运行只允许用全 16 维模型做一个同 seed、同 MLP、同 BC 配置的受控参考；该 checkpoint 不得进入 PPO、主模型选模或回退。消融应同时报告：

- 16 维 teacher-feature BC validation；
- 14 维 no-teacher-feature BC validation；
- 14 维 BC warm start + PPO validation。

## 3. 奖励与 PPO 数学契约

设第 `t` 个动作执行后，当前已调度任务的最大完成时刻为 `C_t`，同实例冻结 HEFT makespan 为 `M_HEFT`。每步奖励固定为：

\[
r_t=-\frac{C_t-C_{t-1}}{M_{HEFT}},\qquad C_0=0
\]

由于 `C_t` 单调不减，整条轨迹满足：

\[
\sum_t r_t=-\frac{C_T}{M_{HEFT}}=-ratio
\]

实现逐 episode 检查该恒等式，绝对误差超过 `1e-9` 立即失败。`gamma` 固定为 `1.0`，避免折扣破坏奖励与主指标的严格对应。

critic 输入为当前合法候选 14 维特征的逐维 mean/max pooling，不读取未来动作或 HEFT 选择。GAE 固定为：

\[
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
\]

\[
A_t=\delta_t+\gamma\lambda A_{t+1}
\]

actor 使用 clipped surrogate：

\[
L^{clip}=E_t\left[\min\left(\rho_t A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\right)\right]
\]

其中 `rho_t` 是新旧 masked policy 概率之比。actor 做梯度上升，critic 对 GAE return 的半平方误差做梯度下降；二者使用独立 Adam 状态和梯度裁剪。每轮记录 policy loss、value loss、entropy、approximate KL、clip fraction 和梯度范数，超过冻结 `target_kl` 时停止本轮剩余 update epoch。

## 4. 数据用途与选模

| 数据 | 允许用途 | 禁止用途 |
| --- | --- | --- |
| train | 生成 HEFT teacher、BC 梯度、PPO rollout/update | 主 checkpoint 最终选择 |
| validation | teacher/reference 诊断、每轮 checkpoint 选择、止损 | PPO 梯度、最终 test 结论 |
| test | P1-A02 完全禁止访问 | teacher、BC/PPO 梯度、超参数、早停、回退、checkpoint 选择 |

每个 seed 的候选 checkpoint 包括 14 维 BC warm start（epoch 0）和全部 PPO epoch。排序键固定为：

1. `failure_count` 更低；
2. validation `mean_ratio` 更低；
3. 完全相同时选更早 epoch。

因此 PPO 退化时会明确选择 warm start，并在结果中记录 `selected_warm_start=true`；不得用 last 冒充 best，也不得删除失败实例。

## 5. 正式配置与停止条件

正式配置为 `configs/stg_ppo.json`：

| 项目 | 值 |
| --- | ---: |
| seeds | 20260717、20260718、20260719 |
| BC epoch | 1 |
| PPO epoch | 2 |
| train episodes / PPO epoch | 120 |
| update epochs | 2 |
| minibatch | 256 transitions |
| gamma / GAE lambda | 1.0 / 0.95 |
| clip ratio | 0.2 |
| actor / value learning rate | 0.0003 / 0.001 |
| entropy coefficient | 0.001 |
| target KL | 0.03 |

本轮 validation 门禁要求每个 seed 同时满足：

- failure count 为 0；
- illegal action count 为 0；
- best validation `mean_ratio ≤ 1.00`。

全部 seed 通过才建议交 B 独立复核。任一 seed 不通过时，结论必须是回退 P1-A01 并只在 validation 上调参；不得开始 task-GNN 或访问 test。

## 6. checkpoint 与输出契约

运行命令：

```powershell
python -m trisched train-ppo --config configs/stg_ppo.json
```

每个 seed 输出：

- 14 维 BC warm-start actor；
- PPO best/last actor checkpoint；
- PPO best/last value checkpoint；
- BC + PPO 训练曲线；
- best/last validation 逐实例诊断；
- validation failure JSONL。

actor checkpoint 可直接由 `MaskedMLPPolicy.load` 加载；value checkpoint 可由 `ValueNetwork.load` 加载。run manifest 对全部构件记录 bytes/SHA-256，并为 actor/value 分别记录不依赖 NPZ 容器元数据的 parameter SHA-256。

总摘要必须记录 3 个 seed 的单独结果、seed mean/std、改善 seed 数、warm-start 回退数、总 failure/illegal action 数、validation 门禁决定和 `test_accessed=false`。

## 7. 当前实现验证与 B 复核要求

A 提交前已完成：

- 微型公开格式数据上的 3-seed、无 test 字节双次端到端复现；
- 子集特征 actor/value checkpoint 保存、加载与合法调度复评；
- GAE 终局 bootstrap 单元测试；
- run manifest 全 artifact 大小和 SHA-256 重算；
- actor log-prob、entropy 和 critic value-loss 的中心差分梯度核对，最大绝对误差分别约 `3.0e-10`、`3.2e-10`、`4.2e-12`；
- 完整回归 `169 passed`。

B 独立复核至少应覆盖：

1. 从用户推送后的不可变实现提交运行远端 7 环境 CI；
2. 使用完全不含 test 原始字节的数据根重新运行正式 3 seeds；
3. 独立推导奖励 telescope、GAE terminal bootstrap 和 PPO clipping 的正负 advantage 分支；
4. 篡改配置重新加入任一 HEFT 二值特征，确认训练前失败；
5. 加载每个 seed 的 best/last actor/value，重算 validation 指标和参数 hash；
6. 核对 best 可能合法回退 epoch 0，且 selection 没有使用 last、test 或 teacher accuracy tie-break；
7. 执行完整测试并重算 run manifest 全部 artifact。

## 8. 已知限制

- 仍是手工候选特征 MLP，没有任务图消息传递；
- 当前只做 3 个开发 seed 和 30 个 validation 场景，不是最终 5-seed test 主结果；
- PPO 每个 epoch 每个 train 场景只采一条轨迹，样本量仍小；
- critic 的 mean/max pooling 无法表达全部 DAG 拓扑；
- 当前结果无论是否达到 ratio 1.0，都不能在公开 test 和最终统计前写成竞赛性能领先。
