# P1-A01 HEFT teacher 与公开 STG 行为克隆基线

- 任务：P1-A01 / P1-02
- 主责：成员 A
- 复核：成员 B
- 日期：2026-07-17
- 状态：已完成（B 独立复核通过）
- 不可变实现提交：`2eda5ae0eeca87affd54f4067ba17d4cdff1f766`
- 被复核远端提交：`180a7130517df425bf53dab945f3760e5bcdba72`
- 独立复核：[P1-A01 独立复核记录](./P1-A01独立复核记录.md)
- 数据契约：`stg-rnc50-hetero-trisched-v1`

## 1. 目标与非目标

本任务在 P1-B01 冻结的公开 STG topology projection v1 上建立一个可重生成的纯行为克隆基线：只从 train 生成 HEFT teacher，validation 只做诊断和 checkpoint 选择，test 原始场景不进入 teacher、梯度更新、早停、超参数或 checkpoint 选择路径。

本任务不实现 PPO/GNN，不访问公开 test，不宣称优于 HEFT。当前 16 维候选特征包含 `is_heft_task` 和 `is_heft_pair`，因此该模型的意义是证明公开数据、teacher、mask、checkpoint 和选模证据链闭合，不是算法创新结果。

## 2. 数据用途硬门禁

`trisched.benchmark.load_frozen_split` 要求调用方同时声明 split 和 purpose；允许矩阵固定为：

| purpose | 允许的 split | 用途 |
| --- | --- | --- |
| `teacher` | train | 生成冻结 HEFT teacher |
| `training` | train | 梯度更新 |
| `model_selection` | validation | 诊断、best checkpoint 选择 |
| `evaluation` | validation、test | 独立评测；不由 `train-bc` 调用 test |

不匹配时在读取对应源 JSON 前以稳定错误码 `split_usage` 失败。`train-bc` 的配置没有训练/选模 split 开关，只固定加载 train 和 validation；正式 run manifest 记录：

```json
{
  "loaded_splits": ["train", "validation"],
  "teacher_split": "train",
  "model_selection_split": "validation",
  "test_accessed": false
}
```

测试还明确尝试把 test 用于 `training` 和 `model_selection`，两者均被拒绝。B 复核时必须再使用一个只含 150 个 train/validation JSON、完全不含 30 个 test JSON 的新 raw root 成功重跑，以证明训练过程不依赖 test 字节。

## 3. 冻结 teacher 契约

对每个 train 和 validation 场景分别运行：

1. 生产 `HeftPolicy`；
2. 不复用生产时间计算的 `independent_heft_schedule`；
3. 生产 validator 与独立 validator；
4. 逐动作核对 task/resource、start/finish 和 makespan；
5. 固定 source/scenario hash、动作序列、trace hash 和完整 schedule hash。

train 输出 `train_teacher_manifest.json`，purpose 为 `behavior_cloning_teacher`；validation 输出 `validation_reference_manifest.json`，purpose 为 `model_selection_reference`，只用于诊断/选模。正式结果：

| 项目 | train teacher | validation reference |
| --- | ---: | ---: |
| Scenario | 120 | 30 |
| 每图任务 | 50 | 50 |
| 总动作 | 6,000 | 1,500 |
| 双实现不一致 | 0 | 0 |
| validator 失败 | 0 | 0 |

train 有序 trace hash 聚合值：

```text
51acb3c9efb00a18571ebe8c7b9723f36035044571f78939ddb40b7f799e4d69
```

train 有序 schedule hash 聚合值：

```text
96859e209bfe9091aac81f6fc705c686ccf29dd4611d775ccf4e22560c23e29c
```

`teacher_failures.jsonl` 为空。任何动作、Scenario 或投影字节变化都会使 trace/source/Scenario hash 校验失败。

## 4. BC 配置与选模规则

固定配置为 `configs/stg_bc.json`：

| 字段 | 值 |
| --- | ---: |
| seed | 20260717 |
| hidden dimension | 32 |
| epoch | 3 |
| learning rate | 0.004 |
| gradient clip | 5.0 |
| failure penalty ratio | 10.0 |

候选特征和 teacher 状态只由 Scenario 与冻结动作轨迹决定，因此先重放一次并物化为只读特征批次；不同 epoch 复用相同状态，避免重复图搜索。策略仍逐动作执行相同 Adam 更新，没有改变原行为克隆目标。

每个 epoch 后只在 validation 运行确定性策略。best 的排序键固定为：

1. failure count 为 0；
2. `validation mean_ratio` 更低；
3. teacher-state action accuracy 更高；
4. 完全相同时选更早 epoch。

交叉熵只作为诊断，不在 ratio 和准确率相同后推动模型继续过度自信。test 不参与任何 tie-break。

## 5. 训练曲线与 validation 结果

| epoch | train CE | train action accuracy | validation ratio | validation teacher accuracy | validation CE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0164209904 | 0.9983333333 | 1.0 | 1.0 | 0.0000155277 |
| 2 | 0.0000041865 | 1.0 | 1.0 | 1.0 | 0.0000005587 |
| 3 | 0.0000001606 | 1.0 | 1.0 | 1.0 | 0.0000000237 |

epoch 1 已达到 30/30 validation 场景与 HEFT 完全一致，因此按冻结规则选为 best；epoch 3 仍作为 last 保留。best 的正式 validation 指标为：

| 指标 | 值 |
| --- | ---: |
| success / failure | 30 / 0 |
| valid schedule rate | 1.0 |
| mean ratio vs HEFT | 1.0 |
| teacher-state action accuracy | 1.0 |
| rollout actions | 1,500 |
| illegal actions / rate | 0 / 0.0 |

`validation_failures.jsonl` 为空。ratio 1.0 只表示复制 HEFT，不允许写成优于 HEFT。

## 6. checkpoint 与可重生成标识

NPZ 文件的 SHA-256 固定本次容器字节；`parameter_sha256` 另按特征 schema、参数名/shape 和 little-endian float64 权重计算，不依赖 ZIP 元数据：

| checkpoint | epoch | NPZ SHA-256 | parameter SHA-256 |
| --- | ---: | --- | --- |
| `bc_best.npz` | 1 | `fa5fa524242d58c46173211bb72ee168e8d7a7302274ff82bbcf67cd174e7394` | `fb3bba126058c3e4053d290e753a065bfa58f7ffe1c64c8ed83b6310b5d15388` |
| `bc_last.npz` | 3 | `1239ac1fd4edfdf8dfe14c0b938c9e90d6bdeff26629b39f1f8c0d7e6bac6dbf` | `343fde145197a0711dd61b9f00d73b975b8a5454da35c403132991f49b9e3f6b` |

best 与 last 参数哈希不同，证明 best 不是把最终 checkpoint 改名。B 复核应以 parameter hash 为跨机器主要判据，并另外记录 NPZ 容器 hash 是否一致。

## 7. 运行清单与测试

正式运行来自干净实现提交 `2eda5ae0eeca87affd54f4067ba17d4cdff1f766`，run manifest 记录 `working_tree_dirty=false`。10 个声明构件的 bytes/SHA-256 已全部重算一致：

- resolved config；
- train teacher 与 validation reference manifest；
- teacher/validation failure JSONL；
- 训练曲线；
- best/last checkpoint；
- validation 逐实例诊断；
- summary。

关键证据 hash：

| 文件 | SHA-256 |
| --- | --- |
| `bc_summary.json` | `3e9d96048c0e73336a9812ffe166fe090a17a0e99b5f13d7328787fad28fcbe3` |
| `bc_run_manifest.json` | `98f5b1d846e353fd0fc8a55a8d3305ab2296655e31d879a4bf239961c653b6d6` |

正式运行在本机约 194 秒完成；该时间只描述复现成本，不作为算法效率结果。完整回归为 `165 passed in 4.90s`。运行产物位于 Git 忽略的 `outputs/p1-a01-stg-bc/`，不提交第三方原始数据或生成 checkpoint。

## 8. 复现命令

先获取并复核 P1-B01 archive，再运行 BC：

```powershell
python scripts/fetch_stg_benchmark.py --offline
python -m trisched train-bc --config configs/stg_bc.json
```

`train-bc` 拒绝写入非空输出目录，避免把两次证据混在一起。需要重跑时应传入一个新的空目录：

```powershell
python -m trisched train-bc `
  --config configs/stg_bc.json `
  --output outputs/p1-a01-reproduction
```

## 9. B 独立复核结论

B 已在远端不可变提交 `180a7130517df425bf53dab945f3760e5bcdba72` 上完成复核，且没有复用 A 的输出：

| 复核项 | 结果 |
| --- | --- |
| 远端与 CI | Windows/Ubuntu × Python 3.10/3.11/3.12，加 openEuler 24.03 LTS SP4，共 7 个 job 全部成功 |
| 无 test 原始字节 | 只保留 120 train + 30 validation，确认 test 0/30；连续两次训练成功 |
| 可重生成 | 10 个核心构件两次逐字节相同，best/last NPZ 和参数 hash 与正式值一致 |
| teacher 抽查 | 固定抽查 10 个 train 场景，生产/独立 HEFT、动作、makespan、hash 和双 validator 全部一致 |
| 用途门禁 | test 用于 teacher、training、model selection 均在读取 JSON 前以 `split_usage` 失败 |
| 篡改注入 | trace hash 篡改返回 `teacher_trace_hash`；非法动作返回 `teacher_illegal_action` |
| checkpoint 复评 | best/last 均为 validation 30/30 成功、ratio 1.0、failure/illegal action 0 |
| 完整测试 | `165 passed in 5.57s`，隔离 worktree 保持干净 |

详细 SHA、抽查索引、run manifest 和结论边界见[独立复核记录](./P1-A01独立复核记录.md)。P1-A01/P1-02 状态改为“已完成（B 独立复核通过）”，下一主任务为 P1-A02 masked PPO。

## 10. 已知限制

- `is_heft_task`/`is_heft_pair` 直接暴露 teacher 决策，100% 复制是预期 sanity baseline，不是可发表贡献；P1-A02 前必须做移除 teacher 二值特征的消融。
- 本任务只有一个训练 seed；多 seed、置信区间、ID/OOD 和效率报告属于 P1-B02/G3。
- validation ratio 为 1.0，尚无优于 HEFT 的证据；test 完全未访问，因此也没有公开 test 结果。
- 输入是 capability-relaxed STG topology projection，不保留 GPU/core/memory 约束。
- 冻结特征批次减少重复图搜索，但仍是逐动作 NumPy 更新；PPO 阶段需要批量化和明确的训练/推理性能报告。
