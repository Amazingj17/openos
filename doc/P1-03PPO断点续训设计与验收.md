# P1-03 PPO 断点续训设计与验收契约

- 任务：P1-03 / P1-A03 前置门禁
- 主责：成员 A
- 复核：成员 B
- 日期：2026-07-17
- 状态：部分完成（A 已提交 staging 事务修复，待 B 从新远端提交重验）
- 前置证据：[P1-A02 Masked PPO 独立复核记录](./P1-A02独立复核记录.md)
- 本轮复核：[P1-03 PPO 断点续训独立复核记录](./P1-03独立复核记录.md)

## 1. 目标与非目标

本任务为公开 STG masked PPO 增加确定性断点续训，使意外退出后的训练能从最后一个完整 PPO epoch 边界恢复，而不是丢弃全部已完成更新或只从推理 checkpoint 猜测优化器状态。

断点续训不得改变 P1-A02 已冻结的 14 维基础输入、奖励、数据 split、随机种子、checkpoint 选择和 warm-start 回退规则。公开 test 继续禁止加载。本轮不实现 minibatch 内恢复、跨代码版本迁移或 task-GNN；P1-A03 必须等本门禁经 B 复核后再开始实现。

## 2. 用户接口与生命周期

首次运行仍使用：

```powershell
python -m trisched train-ppo --config configs/stg_ppo.json
```

训练会在每个 seed 的 epoch 0 和每个完整 PPO epoch 后原子写入：

```text
seed_<seed>_ppo_training_state.npz
```

进程异常退出后，使用相同配置、代码、数据和输出目录恢复：

```powershell
python -m trisched train-ppo --config configs/stg_ppo.json --resume
```

若首次运行使用了 `--output <dir>`，恢复时必须传入同一个 `--output <dir>`。`--resume` 不会创建缺失的输出目录，也不会把旧版只有推理 checkpoint 的目录视为可恢复状态。

## 3. epoch 边界状态

每份状态文件同时保存：

- actor 当前参数、Adam 一阶/二阶矩、Adam step 和采样 RNG；
- critic 当前参数、Adam 一阶/二阶矩和 Adam step；
- PPO 场景排列/minibatch RNG；
- 当前已完成 epoch、完整 history、best selection key 与 best epoch；
- best actor/value 参数，用于恢复 validation 选模和 warm-start 回退；
- `test_accessed=false`、状态格式版本、恢复契约 hash 和状态内容 hash。

仅保存 actor/value 推理参数不足以续训：丢失任一 Adam 动量、RNG、history 或 best 状态，都可能使恢复轨迹和连续训练分叉。因此普通 `*_best_policy.npz`、`*_last_policy.npz` 和 value checkpoint 仍只用于推理/复评，不能充当训练状态。

## 4. 恢复契约与拒绝条件

恢复状态绑定以下输入：

- 原始配置文件 SHA-256 与规范化配置；
- benchmark manifest SHA-256；
- Git 提交/dirty 元数据和全部 `trisched/*.py` 来源 hash；
- seed、14 维特征名和完整 PPO 配置；
- BC warm-start parameter SHA-256；
- train/validation Scenario hash 序列；
- teacher/reference trace 与 schedule 聚合 hash；
- `test_accessed=false`。

恢复前还会核对输出目录内原 `resolved_config.json`、train teacher manifest、validation reference manifest 和空 teacher failure 文件。任一配置、代码、数据、teacher、warm start 或 test 状态变化都必须失败，禁止“尽量加载”或静默重置 Adam/RNG。

状态文件含覆盖 metadata 和全部数值数组的 `payload_sha256`。加载时还检查数组名、shape、有限值、epoch/history 对齐、best epoch、Adam step 与历史 update/minibatch 数是否一致。

## 5. 原子性与中断语义

单个 seed 的状态仍先写入同目录临时文件，刷新用户态缓冲并执行 `fsync`，最后用原子替换发布。整次 `--resume` 另增加 run 级 staging 事务：先将现有证据目录复制到同盘隐藏 staging，所有 seed、summary 和 manifest 只在 staging 中重建；全部成功后将旧目录移为唯一 backup、将 staging 目录交换到正式路径，再清理 backup。若任一校验、训练或写出失败，staging 被清理，正式目录保持调用前字节。

若进程在当前 epoch 中途退出：

1. 上一个完整 epoch 的状态文件保持有效；
2. 当前未完整 epoch 的内存更新不进入恢复点；
3. `--resume` 重新执行该 epoch；
4. 已完成 seed 可直接从最终状态重建输出，未开始 seed 从 epoch 0 正常启动；
5. 最终 run manifest 明确记录 `resume_requested`、`resumed_seeds` 和边界类型；
6. 任一 seed 在恢复预检或训练中失败时，不得让正式输出目录处于“旧 manifest + 新 artifact”的混合状态；
7. resumed run manifest 必须记录 `publication_mode=staging_directory_swap`，非恢复新运行记录 `direct_new_directory`。

训练成功后状态文件保留为可审计 artifact。P1-A02 原实现提交的正式 run 仍是 31 个声明 artifact；加入断点续训后，每个 seed 新增 1 个状态文件，当前 3-seed 新运行应声明 34 个 artifact。

## 6. 稳定错误码

| 错误码 | 含义 |
| --- | --- |
| `ppo_resume_output_missing` | `--resume` 指向缺失或空目录 |
| `ppo_resume_config_read` | 原规范化配置缺失、损坏或不可读 |
| `ppo_resume_config_mismatch` | 当前配置与中断运行不同 |
| `ppo_resume_artifact_read` | 原 teacher/reference 证据不可读 |
| `ppo_resume_artifact_mismatch` | 原 teacher/reference 证据与当前运行不同 |
| `ppo_resume_state_missing` | 已有 PPO 输出但缺少训练状态，或底层恢复未提供状态 |
| `ppo_resume_state_exists` | 非恢复运行试图覆盖已有训练状态 |
| `ppo_resume_state_read` | NPZ/metadata 无法读取 |
| `ppo_resume_state_version` | 状态格式版本不支持 |
| `ppo_resume_state_hash` | 状态内容与内嵌 payload hash 不同 |
| `ppo_resume_state_mismatch` | 状态契约、seed、特征、网络维度或 epoch 范围不匹配 |
| `ppo_resume_state_corrupt` | 数组、RNG、history、best 或 Adam 计数内部不一致 |
| `ppo_resume_state_write` | 原子状态写入失败 |
| `ppo_resume_stage_write` | 无法创建或复制 run 级恢复 staging |
| `ppo_resume_publish` | staging 发布、回滚或 backup 清理失败 |

这些错误沿用结构化 CLI 诊断，退出码为 2。缺失恢复目录在报错后仍不得被创建。

## 7. A 的实现验收

微型公开格式配置使用 3 个 seed、2 个 PPO epoch。测试在 seed `31` 的第二次 `_update_ppo` 前注入异常，确认状态停在 `completed_epoch=1`；恢复后得到：

- `ppo_summary.json` 与未中断运行逐字段一致；
- 三个 seed 的训练曲线 JSON 逐字段一致；
- 三个最终训练状态 NPZ 逐字节一致；
- best/last actor/value 参数 hash、validation 指标和选模结果一致；
- resumed run manifest 记录 `resume_requested=true`、`resumed_seeds=[31]`；
- artifact 数为 34，全部进入 manifest 大小/hash 校验；
- 物理不存在 test JSON，正常路径和错误路径均未访问 test。

负向注入覆盖：

1. 修改 actor learning rate，返回 `ppo_resume_config_mismatch`；
2. 篡改 train teacher manifest，返回 `ppo_resume_artifact_mismatch`；
3. 删除已有 seed 的训练状态，返回 `ppo_resume_state_missing`；
4. 修改状态 metadata 但保留旧 hash，返回 `ppo_resume_state_hash`；
5. 将状态替换为损坏字节，返回 `ppo_resume_state_read`；
6. 对不存在的目录执行 CLI `--resume`，exit 2、返回 `ppo_resume_output_missing` 且不创建目录。

完整回归结果为：

```text
171 passed in 6.88s
```

## 8. B 独立复核要求

B 应从用户推送后的不可变提交完成：

1. 确认远端 Windows/Ubuntu/openEuler 7 环境 CI 全绿；
2. 在无 test 字节的数据根上注入 epoch 中断并恢复，另跑同配置连续对照；
3. 重算 summary、训练曲线、best/last 参数 hash、最终状态文件和 34 个 artifact；
4. 独立核对 actor/value Adam、两套 RNG、history、best state 和 epoch 边界是否完整；
5. 注入配置变化、teacher/reference 变化、状态 hash 变化、NPZ 损坏和缺失状态；
6. 确认 run manifest 如实记录恢复使用情况，且 test 始终未加载。

B 首轮复核确认正常中断/恢复、状态完整性、34-artifact、错误码和无 test 边界，但因多 seed 后置失败暴露 run 级事务性缺陷而未通过。A 已选择 staging 方案修复：自动测试证明 seed 32 状态不匹配，以及 seed 31 已写完后 seed 32 写出异常，均不会改变正式目录或留下事务目录。B 仍须从新的远端不可变提交独立重复正常与失败路径；重新复核前 P1-03 保持部分完成，P1-A03 task-GNN 不进入实现。

## 9. 已知限制

- 恢复粒度是完整 PPO epoch，不是 episode、minibatch 或单次 Adam update；
- 恢复时会重新生成/核对 teacher、冻结状态、特征消融和 BC warm start，节省的是已完成 PPO epoch，不是全部预处理时间；
- 状态只支持相同配置、代码、数据和 warm start，不提供跨版本迁移；
- A 的 staging 修复已在自动测试中保证故障恢复不改变正式证据目录，尚待 B 从新不可变提交独立确认；
- run 级发布依赖 staging、正式目录和 backup 位于同一父目录，以获得同文件系统目录重命名语义；
- 跨平台目录发布使用“旧目录移至 backup → staging 移至正式路径 → 清理 backup”，并在第二步失败时回滚；不宣称主机断电或进程在两次目录重命名之间被强杀时仍是单指令原子交换，此时须保留并人工核对同级 backup/staging；
- P1-A02 的历史 31-artifact 正式结果不会被改写，新格式只影响本提交之后的运行。
