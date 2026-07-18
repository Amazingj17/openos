# P1-A05 Size-OOD 实现与训练前门禁

- 任务：`P1-A05-IMPLEMENT`
- 主责：成员 A
- 复核：成员 B
- 日期：2026-07-18
- 状态：实现与 B 独立复核通过，receipt 已生成；未训练

## 目标与非目标

本任务只实现已经双签的单一候选：保持五个 seed、14 维 MLP、P1-A04 warm start、奖励、优化器、两轮 PPO、每轮 6000 transitions 和 validation 选模规则不变，把每轮 rollout 改为 `60×STG-50 + 30×synthetic-100`。

本阶段不加载正式 checkpoint、不创建优化器、不训练、不评测 development、不访问 public test，也不改变预注册设计。

## 实现组成

- `configs/p1_a05_size_robustness.json`：生产配置，精确绑定预注册、P1-A04 五个 warm-start 字节/参数 hash、五 seed 和两轮 rollout 列表。
- `trisched/p1_a05.py`：严格配置拒绝器、隔离输入物化、去重、dry-run、复核 receipt 校验和正式训练事务。
- `trisched/ppo.py`：为 masked PPO 增加可选的逐 epoch 场景 ID 列表；旧调用不传该参数时行为保持不变。
- `trisched/cli.py`：新增 `prepare-p1-a05`、`dry-run-p1-a05`、`train-p1-a05`。
- `tests/test_p1_a05_implementation.py`：冻结变量注入、rollout 列表、dry-run、训练前拒绝和源码 commit 绑定测试。

## 隔离输入物化

执行：

```powershell
python -m trisched prepare-p1-a05 --config configs/p1_a05_size_robustness.json
python -m trisched dry-run-p1-a05 --config configs/p1_a05_size_robustness.json
```

输出位于被 Git 忽略的 `outputs/p1-a05-training-input/`。物化器使用隐藏 staging 目录，全部校验通过后才原子发布；异常会删除 staging。首次真实执行发现 `synthetic/` 子目录未先创建，事务正确清理了半成品；修复后重新执行成功且无 staging 残留。

实际审计结果：

| 检查项 | 结果 |
| --- | ---: |
| benchmark train / validation | 120 / 30 |
| copied benchmark entries | 150 |
| synthetic-100 | 60 |
| synthetic 唯一 ID / 内容 hash | 60 / 60 |
| P1-A04 warm-start | 5 |
| 与 benchmark train/validation 的 ID/hash 交集 | 0 / 0 |
| 与 120 个 development 场景的 ID/hash 交集 | 0 / 0 |
| public-test 文件 / archive / 禁用文件名 | 0 / 0 / 0 |
| epoch 1 episodes / transitions | 90 / 6000 |
| epoch 2 episodes / transitions | 90 / 6000 |
| checkpoint loaded / optimizer created / training started | false / false / false |

本次本地产物 SHA-256：

- prepared manifest：`3fe388b38325d575198f913c83ed58e63f52389d632eb23dde5de96cde1a96fa`
- rollout dry-run：`b81fde389e73a45449f8caf190af23bd15cd43387b56f2dc0bf8f02551887b2a`
- 生产配置 raw SHA-256：`9c8542efd51d1ced3be3a7a11532d601fb0191513f40a9c4158ad98b1cd267c3`
- 生产配置 canonical SHA-256：`62afb332f73cd790f77c0b9ec0e228c712bb101ad10904ab47abe390a70d3d34`

这些本地产物不提交 Git；B 的复核 receipt 必须绑定其实际复核副本的完整 SHA-256。

## 严格配置拒绝器

生产配置拒绝 missing/unknown 字段，并精确比较预注册和 P1-A04 基础配置。唯一允许的耦合变化是 `episodes_per_epoch: 120 → 90`，其含义由固定 rollout 列表补足到相同的 6000 transitions。以下注入已被专项测试拒绝：

- 新增未经复核变量；
- 修改 gamma 或 MLP hidden dimension；
- 改为 91 episodes 或 6001 transitions；
- 允许 public test；
- epoch 列表数量错误、列表长度错误、ID 重复或未知 ID。

## 正式训练前门禁

`train-p1-a05` 在创建正式输出和加载 checkpoint 前读取 `configs/p1_a05_implementation_review.json`。receipt 必须：

- `task=P1-A05-IMPLEMENT-REVIEW`、`approved=true`、`reviewer=B`；
- 绑定生产配置 canonical hash、预注册 canonical hash、prepared manifest raw hash、dry-run raw hash；
- 把 public test 固定为 `forbidden`；
- 六个冻结 gate 全为 true；
- `approved_source_commit` 为远端不可变实现 commit；
- 正式训练工作树干净，且 `.gitattributes`、`trisched/`、生产配置、依赖声明和 `pyproject.toml` 与 `approved_source_commit` 零差异；允许后续提交只增加 receipt/复核文档。

在 A 的实现提交中 receipt 故意不存在。B 复核前实际执行：

```powershell
python -m trisched train-p1-a05 --config configs/p1_a05_size_robustness.json
```

返回退出码 2 和 `p1_a05_review_missing`；`outputs/p1-a05-size-robustness/` 未创建。测试还用 monkeypatch 证明 checkpoint loader 未被调用，并验证批准 commit 后任何受控训练源码变化都会失败、只增加 receipt 的后续 commit 可通过源码树门禁。B 现已完成独立复核并生成 receipt；包含 receipt 的提交被用户推送确认前仍禁止执行该命令。

## 验证结果

```powershell
python -m pytest -q tests/test_portable_hashing.py tests/test_p1_a05_implementation.py
python -m pytest -q
python -m compileall -q trisched scripts tests
python -m json.tool configs/p1_a05_size_robustness.json
git diff --check
```

- 两个专项文件当前 20 项通过；
- 全量回归当前 256 项通过；
- compileall、严格 JSON 解析和 `git diff --check` 通过；
- 正式训练入口拒绝，未创建 checkpoint/优化器/训练输出；
- public test 未访问、未复制、未物化。

## B 的独立实现复核（已完成）

成员 B 已从远端不可变提交：

1. 在干净 worktree 重跑全量测试和严格 JSON 校验；
2. 重新物化隔离训练根，独立重算 60 个场景 ID、内容 hash 和四类去重交集；
3. 核对 prepared 根只含 train/validation、synthetic、五个 warm-start 和控制文件；
4. 重跑 90/6000 dry-run，并执行配置/rollout/public-test 负向注入；
5. 核对五个 warm-start 原始与参数 hash；
6. 在 receipt 不存在时确认训练入口未创建输出、未调用 checkpoint loader；
7. 只有全部通过，才创建绑定实际远端 commit 和本次产物 hash 的 `configs/p1_a05_implementation_review.json`。

完整结果见[P1-A05 实现独立复核记录](./P1-A05Size-OOD实现独立复核记录.md)。receipt 已写入 `configs/p1_a05_implementation_review.json`，仍须提交并由用户推送。只有 receipt 所在远端提交可复核、且其批准的实现 commit 不变后，才允许启动唯一正式训练。

## 已知限制与停止规则

- synthetic-100 与 development size-OOD 同为 100 tasks，但生成器和结构仍不同；结果只能检验预注册干预，不能宣称纯 size 因果。
- 每个 seed 只运行预注册的两轮；不增加 epoch、不换 seed、不改比例、不临时调参。
- 正式 validation 未改善时允许按冻结选模回退原 warm-start，但不得补跑第二候选。
- development 四切片仍只用于正式训练后的唯一候选评测；size 门禁未通过前继续禁止 public test 和 G3。
