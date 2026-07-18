# P1-B02 OOD 证据路径独立复核记录

- 任务：`P1-B02-OOD1-REVIEW`
- 主责：B
- 复核：A
- 被复核提交：`b22f288271261201c409c4f10adf3d3ec59006f0`
- 日期：2026-07-18
- 结论：**退回**

## 1. 复核边界

A 在用户确认推送后执行 `git fetch origin main --prune`，确认本地 `HEAD`、`origin/main` 和 `FETCH_HEAD` 均指向 `b22f288271261201c409c4f10adf3d3ec59006f0`，随后从该提交建立干净 detached worktree。复核脚本位于 Git 忽略目录，没有导入 `tests/test_p1_b02_ood.py` 或其中任何 fixture/helper。

本轮只复核 development materializer、冻结 manifest、只读 evidence producer、失败语义和 scheduler-only 计时。没有加载 checkpoint，没有运行 HEFT/PPO/GNN 等真实策略，没有启动训练，也没有访问 public test。用于真实物化的数据根被物理缩减为：

```text
train files       0
validation files 30
test files        0
```

## 2. 通过项

### 2.1 来源、完整 payload 与确定性

A 没有调用 B 的 transform helper，而是独立重写以下逻辑并逐字段比较：

1. STG JSON 到 TriSched Scenario 的 topology/duration/data projection；
2. NumPy `PCG64` 下的 100-task OOD-Size 生成；
3. CCR 的 edge data ×2、非对角 bandwidth ×0.5；
4. System 的资源速度 ×`[0.65, 1.0, 1.35]`、非对角 bandwidth ×0.75、latency ×1.5。

结果为 120/120 个完整 Scenario payload 一致，ID 和 content hash 均 120 个且无重复；两次从 `0/30/0` 隔离根运行生产 CLI，各生成 121 个文件，逐字节差异数为 0。冻结 manifest 与新生成 manifest 逐字节一致：

```text
bytes   86559
sha256  d9c6e22ee7991b5659a46b6503a98580e9f3262edc5381bc9a32b8cf44acf2be
```

四个独立重算的 aggregate hash 为：

```text
id_validation fea76e27e4c0a90e1a3067b9367f7e22372c5463e23b65abaf70aa5695010b1b
ood_size      6f525035c5bc07b0a9b6dfd9fc456ac3d2343d234d409c5259ae944967f0da1a
ood_ccr       fef5aa2ce9526a9a91e1479c52f5bfbe30e34d9ce45e11311d231e2f1c6f53c8
ood_system    618942a58ccd2f20a1b97966f492fe00030225a7a02ba28c7f2a8fc54bc26431
```

### 2.2 篡改拒绝与无 test 边界

A 独立注入 8 类异常，均得到预期稳定诊断：

| 注入 | 实际 code | 结果 |
| --- | --- | --- |
| 场景文件追加字节 | `ood_file_hash` | 通过 |
| `test_accessed=true` | `ood_test_gate` | 通过 |
| ID provenance scenario hash 漂移 | `ood_source` | 通过 |
| 场景路径改为 `../escape.json` | `ood_manifest` | 通过 |
| 契约 CCR scale 漂移 | `ood_manifest` | 通过 |
| 原始 validation 文件字节漂移 | `ood_source` | 通过，且无输出目录 |
| development 契约请求 test | `ood_contract` | 通过，且无输出目录 |
| 目标 materialization 目录已存在 | `ood_output_exists` | 通过 |

生产 CLI 在只有 30 个 validation 文件、完全没有 train/test 文件时成功，证明当前 materializer 不依赖 public-test 字节。manifest 和 evidence 均保持 `test_accessed=false`、`public_test_loaded=false`、`training_started=false`。

### 2.3 scheduler-only 计时和基本失败路径

A 构造独立 2 场景 × 4 切片 × 2 runner 的微型契约。fake runner 每次只推进时钟 4 ms；生产 validator 与独立 validator 各额外推进 250 ms。16 次 scheduler 调用的 evidence runtime 全部为 4 ms，事件序列均为：

```text
clock(start) -> runner.schedule -> clock(stop)
             -> production validator -> independent validator
```

因此 producer 自己的两个 validator 确实位于停止计时之后。普通非参考 timeout 保留 8 条 penalty=`10.0` 记录；返回后被 validator 判非法的 8 条结果均记录 `scheduler_invalid_schedule` 和 `illegal_action_count=1`；HEFT timeout 返回 `ood_reference_failed` 且不写 evidence；runner name 错配返回 `ood_runner_provider`。成功 evidence 可直接通过自动报告器。

## 3. 阻塞缺陷

### R1：runner 内部发现的非法调度被漏计

`_schedule_only` 只在 runner 正常返回 `ScheduleResult`、随后由 producer validator 拒绝时把 `illegal_action_count` 置为 1。若 runner 在调用内部先发现非法行为并抛异常，当前异常分支固定保留 `illegal_action_count=0`。

A 用两条生产相关路径复现：

1. 模拟 `ExternalProcessScheduler` 抛出 `SchedulerAdapterError(code="scheduler_invalid_schedule")`：8 条记录保留正确 error code，但 `illegal_action_count` 全部为 0；自动报告在四个切片中均显示 `failure_count=2`、`illegal_action_count=0`、`illegal_record_count=0`。
2. 使用真实 `PolicySchedulerRunner` 包装一个返回 `(-1, -1)` 的非法动作策略：8 条记录被降级为 `scheduler_execution_failed`，`illegal_action_count` 仍全部为 0。

这会使正式报告把已明确发生的非法调度写成“零非法”，与 P1-B02 的失败/非法分项统计不一致。虽然 failure 本身仍会阻止主结果发布，但错误的 illegal 数字不能作为正式证据接受。

修复验收要求：

- adapter 已给出 `scheduler_invalid_schedule` 时，failure 行必须记录非零 `illegal_action_count`；
- 进程内非法动作必须使用稳定的类型化诊断，不得降级为泛化的 `scheduler_execution_failed`；
- 新增两条生产路径测试，并验证自动报告中的 `illegal_action_count/illegal_record_count`。

### R2：错误返回类型泄漏未结构化的 `AttributeError`

若一个名称正确的非参考 runner 返回 dict 等非 `ScheduleResult` 对象，计时结束后代码直接访问 `result.policy_name`，实际抛出：

```text
AttributeError
```

整次 evidence 生产因此中断，既没有稳定 `OODWorkflowError`，也没有按冻结规则把该非参考策略实例保留为 penalty failure。这与已实现的“policy name 错配记为 `scheduler_invalid_response`”边界不一致。

修复验收要求：停止计时后先检查返回对象类型；非参考 runner 的错误类型必须成为 `scheduler_invalid_response` penalty 行，HEFT 的同类错误仍通过 `ood_reference_failed` 阻断。必须补充正常/非参考/HEFT 三条测试。

## 4. 回归与远端证据

干净 detached worktree 中：

```text
Black check   3 files unchanged
compileall    pass
pytest        227 passed in 14.13s
git diff      clean
```

远端 [Actions run 29624725554](https://github.com/Amazingj17/openos/actions/runs/29624725554) 对提交 `b22f288...` 成功：Windows/Ubuntu × Python 3.10/3.11/3.12 与 openEuler 24.03 LTS-SP4 共 7 个 job 全部通过，7 个 evidence artifact 均存在。现有 CI/227 项测试没有覆盖 R1/R2，因此远端全绿不改变本次退回结论。

Git 忽略目录中的独立证据 hash：

| 证据 | SHA-256 |
| --- | --- |
| `audit_static.py` | `461d6657a008a0955264b824ddbd3ca7853e1c80e7fef5bc7430f1f827e26f28` |
| `static_audit_report.json` | `04f9a165ee7b5b5773e59e6d923b355f8f4bf72757fdb389fdb98243bdac57ff` |
| `audit_behavior.py` | `823ff193f161c3705c78729407c5f1a05e2ca022ec666ab8d840fa0ef630999c` |
| `behavior_audit_report.json` | `879d018acb45708d579b2a309deb4635252133a6249417245da18024e5a3cb5d` |
| adapter 非法 evidence | `7d788f148b4ea3afee8290398e592415b0ab0a3682896458b9056e29aa649d7a` |
| 进程内非法动作 evidence | `47112859e36b23cea2cafb40a078c799db7585fc7f2170b56bc70d64890ffc94` |

## 5. 结论与下一步

OOD 场景生成、冻结 hash、无 public-test 字节路径、篡改拒绝和 producer 外层计时边界均通过；但 R1 会发布错误的非法统计，R2 会让非参考 runner 协议错误逃逸为未结构化异常。因此 A **退回 P1-B02-OOD1**，P1-B02 继续保持“部分完成”。

下一步由 B 只修复 R1/R2 并补生产路径测试，不重新物化场景、不运行 5-seed development/OOD、不访问 public test、不启动训练。用户推送修复提交后，A 从新的不可变提交二次复核；通过后才允许进入 5-seed artifact 和正式 development/OOD 运行。

## 6. B 修复候选（待 A 二次复核）

B 已在基线 `70472775ac2e6e0d1f1cf37bf151575691e215d1` 上完成两项定点修复：类型化进程内非法动作并统一映射为 `scheduler_invalid_schedule`，evidence 对该错误固定计入一次 illegal；在访问返回字段前验证 `ScheduleResult`，错误类型转为 `scheduler_invalid_response` penalty，HEFT 保持 fail-fast。新增测试覆盖 adapter、真实进程内 runner、自动报告聚合、非参考错误对象与 HEFT 错误对象。

B 的候选验证为 OOD 专项 `13 passed`、报告器合并 `41 passed`、兼容回归 `26 passed`、全量 `229 passed`，Black check 与 `compileall` 通过。以上是修复方自测，不改变本记录第 5 节的独立退回结论；用户推送后仍须由 A 从新不可变提交复现 R1/R2 并签字。materializer/manifest 未变化，真实策略、5-seed OOD、public test 和训练仍未运行。
