# P0-B03 CPOP 与小图精确 Oracle 设计及测试报告

- 日期：2026-07-16
- 主责：成员 B
- 复核：成员 A
- 状态：已通过（A 复核）

## 1. 目标与非目标

本任务在统一 Scheduler 接口中增加 CPOP，并建立一个不依赖生产调度环境的小图精确 solver，用于回答“启发式结果距离当前模型下的全局最优值有多远”。交付包括：

- CPOP upward/downward rank、优先级、关键路径和关键资源选择；
- 可处理插入式时间线及非对称通信的精确分支定界 solver；
- 10 个小图的冻结 CPOP 轨迹、解析下界和已证明最优 makespan；
- CPOP 接入 validation/test 的统一 CSV、摘要指标和示例调度；
- 正常、边界、资源限制和搜索预算失败路径测试。

本任务不把精确 solver 用于训练规模实例，也不宣称 CPOP 或当前学习策略已经稳定优于 HEFT。公开 benchmark、PPO/GNN 和主结果统计仍须等待 G1 正确性门禁通过。

## 2. CPOP 定义与确定性规则

对任务 `i`，平均执行代价为所有资源执行时间的均值；边 `i→j` 的平均通信代价为所有不同资源有向组合的均值。CPOP 使用：

```text
rank_u(i) = avg_exec(i) + max(avg_comm(i,j) + rank_u(j))
rank_d(i) = max(rank_d(p) + avg_exec(p) + avg_comm(p,i))
priority(i) = rank_u(i) + rank_d(i)
```

根任务的 `rank_d` 为 0。为支持多根、多出口和断连子图，本实现使用以下冻结规则：

1. 从 `rank_u` 最大的根开始，任务 ID 用作平局裁决；
2. 每一步沿 `avg_comm + rank_u(child)` 最大的后继继续，形成一条确定关键路径；
3. 关键资源最小化该路径全部任务的执行时间总和，资源 ID 用作平局裁决；
4. 每次从 ready tasks 中选择 `priority` 最大者；关键路径任务固定到关键资源，其他任务选择最早完成资源；
5. 所有剩余平局依次按任务 ID、资源 ID 裁决。

这一定义位于 `trisched/policies.py`，通过 `CpopPolicy.name == "cpop"` 使用现有 `run_policy`，因此与 HEFT、Greedy-EFT、Random 和学习策略共享动作合法性、生产 validator 及独立 validator。

## 3. 精确 solver 与最优性证明边界

`trisched/exact.py` 不导入 `trisched.env` 或生产策略。它直接读取 `Scenario` 原始字段，独立计算执行/通信时间并维护资源区间。

搜索枚举每个状态下所有“ready task × resource”选择，并把任务放入该资源的最早可行插入位置。当前模型无抢占、无链路竞争，目标仅为 makespan；在给定任务资源分配及资源顺序后，故意延迟任务不会改善 makespan，因此至少存在一个最优 active schedule，枚举上述选择可以覆盖它。

分支定界过程为：

1. 用独立 HEFT oracle 取得初始可行上界；
2. 深度优先枚举所有 ready task 和资源选择，按候选完成时间排序以尽早改进上界；
3. 当前部分调度 makespan 已不小于最佳上界时剪枝，因为继续加入任务不可能降低它；
4. 搜索空间穷尽后返回 `proven_optimal=true`、最优调度、解析下界、探索状态数和剪枝状态数。

解析下界取以下两项最大值：忽略通信且所有任务使用最快资源的最长依赖链，以及总工作量除以资源总速度。它用于结果 sanity check；真正的最优性结论来自有限搜索空间穷尽，而不是要求解析下界与最优值相等。

为防止误用于大实例，默认硬限制为 8 个任务和 2,000,000 个搜索状态。超限会抛出 `ExactSolverLimitError`，不会返回未经证明的近似值。

## 4. 十个冻结小图结果

下表所有 `OPT` 均由精确搜索穷尽证明；gap 定义为 `heuristic / OPT - 1`。

| 案例 | 解析 LB | OPT | HEFT | HEFT gap | CPOP | CPOP gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G01 chain | 3.0000 | 3.0000 | 3.0000 | 0.00% | 3.0000 | 0.00% |
| G02 fork/tie | 3.3333 | 5.0000 | 5.0000 | 0.00% | 5.0000 | 0.00% |
| G03 diamond | 3.5000 | 4.5000 | 4.7500 | 5.56% | 5.7500 | 27.78% |
| G04 single resource | 5.0000 | 5.0000 | 5.0000 | 0.00% | 5.0000 | 0.00% |
| G05 insertion gap | 6.4545 | 7.1000 | 13.0000 | 83.10% | 13.0000 | 83.10% |
| G06 fast download | 5.0000 | 6.5000 | 6.5000 | 0.00% | 7.0000 | 7.69% |
| G07 fast upload | 5.0000 | 5.5000 | 5.5000 | 0.00% | 5.5000 | 0.00% |
| G08 identical resources | 2.0000 | 2.0000 | 2.0000 | 0.00% | 2.0000 | 0.00% |
| G09 latency only | 5.0000 | 6.0000 | 6.0000 | 0.00% | 6.0000 | 0.00% |
| G10 disconnected | 4.0000 | 5.0000 | 5.0000 | 0.00% | 6.0000 | 20.00% |

关键结论不是“某个启发式总是更好”，而是：G03 和 G05 已形成两个可复现的 HEFT 非最优反例；CPOP 在部分图上与 HEFT 持平，在 G03、G06 和 G10 更差。这些差异为后续策略学习和失败案例分析提供了可靠标签。

## 5. 统一评测结果

`python -m trisched pipeline --config configs/smoke.json` 已将 CPOP 写入 validation/test 逐实例 CSV、示例 schedule 和 `summary.json`。本地 smoke 结果为：

| split | 样本数 | CPOP mean ratio vs HEFT | 95% CI | win rate | tie rate | 合法率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 10 | 1.0170 | [0.9881, 1.0459] | 20% | 30% | 100% |
| test | 20 | 0.9878 | [0.9621, 1.0136] | 65% | 10% | 100% |

test 均值略低于 1，但置信区间跨过 1，validation 均值也高于 1；数据仍是单次小规模合成集。因此该结果只能证明 CPOP 已正确接入并呈现互补行为，不能作为“显著优于 HEFT”的获奖主结论。

## 6. 测试、产物与复现

从仓库根目录运行：

```powershell
python -m pytest -o addopts="" -q
python -m trisched pipeline --config configs/smoke.json
```

2026-07-16 本地实测：

```text
94 passed
done: test mean_ratio=1.0000 (lower is better; HEFT=1.0)
```

新增或关键变更产物：

| 文件 | 用途 | SHA-256（提交前工作树） |
| --- | --- | --- |
| `trisched/policies.py` | CPOP 分析和策略 | `29cb15a4d5ad98118efc7a33915ff6e3e876f7f7f23bd76455bc015789ac67f7` |
| `trisched/exact.py` | 小图精确分支定界 solver | `be2fc8e0aca964ce29cb7b01d071ee01780386036f48b321c096724e710c0095` |
| `tests/fixtures/golden/optimal_cases.json` | 10 例 CPOP 与 optimum/bound 冻结值 | `8767638dd491a051721b3c13f3ff8f7ce011a79f0a96fd561aa4530c1a40cfa1` |
| `tests/test_cpop_exact.py` | 排名、轨迹、最优性及失败路径测试 | `c2ca33949107a10da7f7c1faaf99384e236d7e3703563773f2706019532223a7` |

实现提交：`424e53a`（`feat(scheduling): [P0-B03] add CPOP and exact small-DAG oracle`）。

## 7. 已知限制、回退与 A 的复核要求

- 精确 solver 的指数复杂度是设计事实，只允许用于小图测试，不接入常规 validation/test。
- CPOP 的多根/多出口规则是本项目的确定性扩展；与外部论文或实现比较时必须明确该规则。
- 解析下界忽略通信和资源不可分割性，可能较松，但不会高估最优 makespan。
- 当前没有使用外部 CP-SAT/MILP 包；分支定界 solver 通过穷尽 active schedules 给出同等用途的可信小图最优参考。
- 若 solver 达到任务或状态上限，任务必须明确失败，禁止把 HEFT 上界伪装成 optimum。

成员 A 已使用独立的“资源分配 × 资源顺序 × 约束图最长路”方法复核 G03、G05、G06 和 G07，四例最优值及 CPOP 轨迹全部一致；超限路径、统一 evaluator 和六环境 CI 也已通过。完整证据见[P0-B03 独立复核记录](./P0-B03独立复核记录.md)。
