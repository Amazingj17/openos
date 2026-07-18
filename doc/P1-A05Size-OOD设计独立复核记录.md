# P1-A05 Size-OOD 设计独立复核记录

- 任务：`P1-A05-DESIGN-REVIEW`
- 原主责：成员 A
- 独立复核：成员 B
- 日期：2026-07-18
- 被复核远端提交：`5857931260ce4d5bb6e93e858a8c7e49425aab7f`
- 结论：设计通过；只授权进入 `P1-A05-IMPLEMENT`，尚不授权训练

## 1. 复核结论

B 同意 A 提交的唯一设计候选：每个 PPO epoch 保持 6000 transitions 不变，只把 rollout 场景源从 120 个 50-task STG 场景改成 `60×50-task STG + 30×100-task synthetic`，即 `3000+3000` transitions。五个训练 seed、两个 epoch、14-D masked MLP、P1-A04 warm start 字节、奖励、PPO/GAE、优化器、超参数和只用冻结 ID validation 选 checkpoint 的规则均保持不变。

本次通过只说明设计满足“单一干预、预算匹配、可证伪、停止规则明确”的预注册要求。它不证明该候选一定改善 size-OOD，也不构成训练许可。以下实现门禁尚未完成：

1. 物化 60 个 synthetic-100 训练场景，并按场景 ID 和去除展示 ID/seed 后的内容 hash 与 benchmark train/validation、全部冻结 development 场景做零重叠检查；
2. 在不加载 checkpoint、不创建优化器的 dry-run 中证明每 epoch 恰为 90 episodes、6000 transitions；
3. 实现生产配置拒绝器，禁止 rollout source 及其耦合 episode count 之外的变量变化；
4. 证明最终训练根物理不存在 public-test JSON、archive 或其他携带 public-test 字节的文件；
5. 由 B 从新的不可变实现提交复核以上四项，复核通过后才能执行唯一一次正式训练。

因此 `G3`、public test 和正式训练继续阻塞；`P1-B03-PORTABLE-HASH` 仍是 B 的系统主任务。

## 2. 隔离环境与输入边界

B 在用户推送后确认：

```text
HEAD        = 5857931260ce4d5bb6e93e858a8c7e49425aab7f
origin/main = 5857931260ce4d5bb6e93e858a8c7e49425aab7f
```

随后从该提交建立 detached worktree `outputs/p1-a05-b-design-review-worktree`。隔离目录只导入以下未跟踪输入：

| 输入 | 文件数 | 用途 |
| --- | ---: | --- |
| P1-B02 development 场景与 manifest | 121 | 只读重算 ID/size 场景描述量 |
| P1-B02 development evidence | 1 | 只读重算性能记录 |
| P1-A04 training summary | 1 | 核对 checkpoint 选择元数据 |
| 合计 | 123 | 无 checkpoint、无 public-test 命名文件 |

`development-evidence.json` 声明 `test_accessed=false`；场景 manifest 声明 `test_accessed=false`、`public_test_materialized=false`。B 的独立脚本检查上述三个输入根，public-test/claim-test 命名数据文件为 0。本轮未导入 checkpoint，未调用训练入口、优化器或 public-test gate。

## 3. Detached 回归与 A 报告重建

B 在 detached worktree 执行：

```powershell
python -m pytest
python -m black --check scripts/analyze_p1_a05_size_robustness.py tests/test_p1_a05_preregister.py
python -m compileall -q trisched scripts tests
python scripts/analyze_p1_a05_size_robustness.py
```

结果为：

```text
pytest:     236 passed in 13.29s
Black:      2 files would be left unchanged
compileall: passed
analysis:   passed in 42.5s
```

重建的 `root_cause_analysis.json` SHA-256 为 `987b312d73e0e35e124b34bd917f29aa4a97b762e6bdb1c2c0ae2f07c245f41d`，与预注册绑定值完全一致。

## 4. 独立 JSON/hash 重算

B 另写纯标准库审计脚本，不导入 A 的分析模块，也不调用 `trisched` 的场景、报告或训练函数。脚本直接解析 JSON，独立实现 canonical JSON、LF 归一化、场景内容 hash、DAG 最长路径、性能聚合和设计约束检查。

| 对象 | 独立 SHA-256 | 结果 |
| --- | --- | --- |
| 预注册 canonical JSON | `9ab60842d87e6d906ab18ad135cda2a372be569ad39b28ce6f17c60f1ff64647` | 一致 |
| A 分析脚本 LF 归一化 | `9ca6f0f029d3e947c2be5a17dda1627273c29db72713d753e5eef523b4e354cf` | 一致 |
| A 根因报告原始字节 | `987b312d73e0e35e124b34bd917f29aa4a97b762e6bdb1c2c0ae2f07c245f41d` | 一致 |
| development evidence 原始字节 | `40d781cad7dff39553ba0c027b686e5710d19ceed62fa3bf7f0471a133feb744` | 一致 |
| evidence records canonical JSON | `0cf814874381d2ad392b3ae193fe11726949eefed72500a1f45433953fa8f8dd` | 一致 |
| P1-A04 training summary 原始字节 | `3d06ecd6e211fb5a5731c9138ba15cd672ad18b06f3fadb9d1ad4bc6420e3178` | 一致 |
| development materialization manifest | `d9c6e22ee7991b5659a46b6503a98580e9f3262edc5381bc9a32b8cf44acf2be` | 一致 |

B 还逐个重算 30 个 ID 和 30 个 size 场景的文件 SHA-256 与去除展示 ID/seed 后的内容 hash，60/60 全部匹配 manifest。

## 5. 性能、horizon 与复合偏移重算

### 5.1 性能与 warm-start 反证

| 策略 | size 记录 | mean ratio | win/tie/loss | failure/illegal |
| --- | ---: | ---: | ---: | ---: |
| masked MLP | 150 | 1.568301894088764 | 0/0/150 | 0/0 |
| task-GNN | 90 | 1.573521826400606 | 0/0/90 | 0/0 |

P1-A04 summary 中只有 seed `20260718` 按冻结 ID 规则选择 epoch 0 warm start；该 seed 在 size-OOD 的独立均值仍为 `1.462685574891456`。因此不能把 size 失败归因于“PPO 把 BC 破坏”，也没有证据支持重开已止损的 task-GNN。

### 5.2 场景描述量

| 描述量 | ID validation | size-OOD |
| --- | ---: | ---: |
| 场景数 | 30 | 30 |
| task count 均值 | 50 | 100 |
| 根节点数均值 | 11.266667 | 1.000000 |
| 最长路径 task 数均值 | 9.600000 | 26.033333 |
| 非对角 bandwidth 全局范围 | 100–10000 | 3.098668–13.408821 |
| 非对角 latency 场景均值的均值 | 0 | 0.371052 |

训练 horizon 50 到 size horizon 100 的倍率为 2，但 size 切片还同时改变 generator、根节点结构、关键路径、workload、bandwidth 和 latency。本复核只支持“复合分布与轨迹偏移”表述，不支持纯 size 因果声明。

## 6. 单一干预与预算复核

| Epoch | STG rollout | Synthetic rollout | Episodes | Transitions |
| ---: | ---: | ---: | ---: | ---: |
| 1 | indices 0–59，60×50=3000 | seeds 20261001–20261030，30×100=3000 | 90 | 6000 |
| 2 | indices 60–119，60×50=3000 | seeds 20261031–20261060，30×100=3000 | 90 | 6000 |

五 seed×两 epoch 的形式预算为 60,000 transitions，与 P1-A04 完全相同。synthetic 训练 seed 与冻结 size-OOD seeds `20260801–20260830` 不重叠。预注册仍只允许一次正式运行；若成功标准不满足，必须保留 P1-A04 模型并发布负结果，不允许基于 development 结果继续换 seed 或调参。

## 7. 负向故障注入

B 的独立设计 validator 先接受未修改候选，再逐项注入以下错误：

| 注入 | 结果 | 拒绝原因 |
| --- | --- | --- |
| synthetic seed 改为冻结 size seeds | 拒绝 | 训练 seed 与 size-OOD 重叠 |
| epoch 1 改为 91 episodes | 拒绝 | episode count 变化 |
| epoch 1 改为 6001 transitions | 拒绝 | transition budget 变化 |
| `gamma` 改为 0.99 | 拒绝 | 冻结 PPO 字段变化 |
| masked MLP 改为 task-GNN | 拒绝 | 模型变化 |
| public test 改为 allowed | 拒绝 | public test 未被禁止 |
| 只读分析脚本注入 `train_ppo...` 调用 | 拒绝 | 分析路径出现训练/优化调用 |

结果为 7/7 拒绝。该结果验证的是设计合同；它不能替代 `P1-A05-IMPLEMENT` 中必须落地的生产配置拒绝器、场景物化去重和 dry-run。

## 8. 授权、阻塞项与下一步

本复核将 `P1-A05-SIZE-ROBUSTNESS-DESIGN` 标记为已完成（B 独立复核通过），并授权 A 开始 `P1-A05-IMPLEMENT`。机器预注册中的 `status=design_candidate_pending_b_review` 是被签字的不可变设计内容，为保持 canonical hash 不修改；实际审批状态以本记录和任务日志为准。

下一步按以下顺序执行：

1. 用户推送本复核提交；
2. A 只实现 synthetic materializer/manifest、内容去重、90/6000 dry-run、生产配置拒绝器和物理无 public-test 根证明，不开始训练；
3. B 从新的不可变实现提交执行 `P1-A05-IMPLEMENT-REVIEW`；
4. 只有实现复核通过，A 才能按预注册执行唯一一次五 seed 正式训练；
5. B 同时继续 `P1-B03-PORTABLE-HASH`，不得改写历史 evidence 的原始 hash。

在第 3 步完成前，不创建正式 checkpoint、不运行 P1-B02 四切片新模型评测、不启动 G3、不访问 public test。
