# P0-09 MVP 复现与失败分析

- 任务：`P0-09`
- 主责：成员 A
- 复核：成员 B
- 日期：2026-07-18
- 状态：主责复现完成，等待远端不可变提交复核

## 复现范围

本任务审计最早的 synthetic smoke：16 维 masked MLP 先模仿 HEFT，再执行 5 轮 episodic REINFORCE。它只验证接口、训练、checkpoint、统一 evaluator 和产物闭环，不是当前正式 P1 模型，也不构成公开 test 证据。

`configs/smoke.json` 的 `test` 是按固定 seed 即时生成的 20 个 synthetic 场景；它与 P1-B02 契约中始终禁止访问的公开 STG test 不是同一数据集。

## 复现命令

```powershell
python -m trisched pipeline `
  --config configs/smoke.json `
  --output outputs/p0-09-mvp-audit

python -m trisched evaluate `
  --config configs/smoke.json `
  --checkpoint outputs/p0-09-mvp-audit/masked_mlp.npz `
  --split validation `
  --output outputs/p0-09-checkpoint-audit

python -m trisched validate-scenario `
  --input tests/fixtures/invalid/scenario_cases.json
```

最后一条命令预期返回退出码 2 和结构化 `missing_field`，用于证明错误路径可诊断，不应改成成功。

## 实际结果

执行环境：Windows、Python 3.11.7、NumPy 1.26.4；源码提交 `2ab2c66f1317637d41adb4f28426ee3c726fbc5c`，运行时工作树干净。

### 训练轨迹

| 阶段 | 结果 |
| --- | ---: |
| imitation epoch 1 accuracy | 0.977564 |
| imitation epoch 2/3 accuracy | 1.000000 / 1.000000 |
| imitation epoch 3 cross-entropy | 0.000423 |
| REINFORCE train ratio epoch 1 | 1.003505 |
| REINFORCE train ratio worst epoch 3 | 1.052097 |
| REINFORCE train ratio epoch 5 | 0.999228 |

REINFORCE 的训练采样 ratio 曾恶化后回到约 1，但确定性策略最终仍完全复制 HEFT。训练轨迹不是冻结评测，不能把 epoch 5 的 `0.999228` 写成泛化提升。

### 冻结 synthetic smoke 评测

| 策略 | validation mean ratio | synthetic test mean ratio | test win/tie/loss | 失败 |
| --- | ---: | ---: | ---: | ---: |
| HEFT | 1.000000 | 1.000000 | 0/20/0 | 0 |
| CPOP | 1.017014 | 0.987822 | 13/2/5 | 0 |
| Greedy-EFT | 1.082257 | 1.059155 | 5/0/15 | 0 |
| Random | 3.360904 | 2.900873 | 0/0/20 | 0 |
| masked MLP | 1.000000 | 1.000000 | 0/20/0 | 0 |

checkpoint-only validation 复评再次得到 masked MLP `mean_ratio=1.0`，且没有重新训练。非法 fixture 返回：

```json
{"valid": false, "error": {"code": "missing_field", "path": "$.bandwidth", "message": "missing required field 'bandwidth'"}}
```

### 关键产物

- `run_manifest.json`：绑定代码 commit、运行时、配置、三 split、checkpoint 和 10 个声明产物；
- checkpoint SHA-256：`7fb644af6ae06530fdac556fd07fc4281a560646d16c2ab5f00d2641fbd8f2b7`；
- `summary.json` SHA-256：`1a75f6e7e3e8692c38405c0819e5acb22eaa05eef9577c9bac381c3e27cab568`；
- `training_history.json` SHA-256：`f832cdc64f8bd3e343660b758ffb7afb54507534526043e400813ff2f08194b5`；
- validation/test failure JSONL 均为空文件，SHA-256 为标准空文件 hash `e3b0c442...b855`。

产物位于被 Git 忽略的 `outputs/p0-09-*`，不进入源码提交。

## 失败根因与处理结论

1. **学习结果没有领先 HEFT。** 模仿精度达到 1，16 维输入还直接包含 `is_heft_task/is_heft_pair`；最终策略只是教师复制，不是自主发现更优规则。
2. **legacy REINFORCE 方差大且不稳。** 中间 epoch 的训练 ratio 明显恶化，终局奖励样本效率低；该路径只保留 smoke，不用于正式结果。
3. **样本与图规模太小。** 24/10/20 个场景、8–16 tasks、3 resources 只能发现工程错误，不能支持比赛性能结论。
4. **synthetic test 可重复使用。** 因为它是 smoke 而非一次性公开 test，不能用其数值冒充冻结 benchmark 泛化证据。
5. **正态近似 CI 不适合作为最终统计。** 正式 P1-B02 已改为五 seed、分层配对 bootstrap、逐实例失败计分和冻结 OOD。

这些失败已经在后续主路中分别处理：去除 HEFT 决策位、BC warm-start 后使用 clipped PPO/GAE、冻结公开 STG split、五 seed、独立 validation 选模和 ID/OOD 契约。legacy smoke 不删除，因为它仍是最快的一键闭环和演示入口。

## 完成判据与复核

成员 B 须从远端不可变提交重跑 pipeline 和 checkpoint-only validation，核对上述非时延数字、hash、结构化失败以及“synthetic smoke 不等于 public test”的边界。复核通过后才把 P0-09 标为已完成。
