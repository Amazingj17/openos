# TriSched MVP

[![CI](https://github.com/Amazingj17/openos/actions/workflows/ci.yml/badge.svg)](https://github.com/Amazingj17/openos/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

赛题十五“基于深度强化学习的云-边-端异构计算资源管理调度方法”的最小可执行实现。

它已经形成一个完整闭环：

- 参数化生成任务 DAG、云/边/端资源与通信矩阵；
- 使用插入式时间线模拟执行与跨节点通信；
- 通过统一接口运行 HEFT、CPOP、Greedy-EFT、Random 和学习策略；
- 对所有“就绪任务 × 资源”候选执行严格合法动作掩码；
- 使用不复用环境时间计算的第二套 HEFT oracle 和 validator 交叉检查调度；
- 用小图精确分支定界 solver 冻结 optimum/bound，量化启发式最优性 gap；
- 保留 HEFT 行为克隆 + episodic REINFORCE 作为 legacy smoke，并在公开 STG 路径实现 clipped PPO + GAE；
- 在公开 STG 冻结 split 上生成生产/独立 HEFT 双签 teacher，并只用 validation 选择 BC best checkpoint；
- 在 PPO 主路径删除直接暴露 HEFT 决策的两项二值特征，使用 3 个 seed 和 BC warm-start 回退做 validation 选模；
- 用机器可读契约冻结 ID/OOD、5-seed、失败/时延、配对 bootstrap 和一次性公开 test 工作流；
- 一条命令完成训练、验证、测试并产出 `summary.json`。

这是用于验证赛题接口、环境正确性和实验流程的 MVP，不是最终获奖模型。P1-A02 masked PPO 已由成员 B 在无 test 字节的数据根上独立复跑并复核通过；P1-03 首轮发现的失败恢复 manifest 破坏已由 staging 目录事务修复，并通过 B 的第二轮独立复核。P1-A03 task-GNN 正式 validation 的点估计改善但配对 bootstrap CI 跨 0，B 已从远端不可变提交独立复核并确认按冻结规则保留 MLP，P1-A03 已关闭。P1-B02 的契约/聚合器、OOD materializer、失败语义和 producer/report hash 互操作均已通过双人复核，masked MLP 五 seed 包也已由 B 在物理无 test 根正式预算复跑通过。B 已生成正式 development ID/OOD 候选：2040 条记录全部成功、零非法动作，但 masked MLP 的 size-OOD ratio 为 `1.568302`，自动门禁 `release_publishable=false`。候选正等待 A 从不可变提交独立复核；public test 仍未加载。

## 快速运行

要求 Python 3.10+。唯一运行依赖是 NumPy。

```powershell
python -m pip install -r requirements.txt
python -m trisched pipeline --config configs/smoke.json
```

复现实验或 CI 应使用锁定依赖：

```powershell
python -m pip install --no-deps -r requirements-lock.txt
python -m pip install --no-build-isolation --no-deps .
```

默认在 `outputs/smoke/` 生成：

```text
summary.json                     主指标与全部策略汇总
validation_per_instance.csv     验证集逐实例结果
test_per_instance.csv           测试集逐实例结果
masked_mlp.npz                  可加载的策略参数
training_history.json           模仿学习和 REINFORCE 训练记录
dataset_manifest.json           场景 ID、hash 和规模清单
resolved_config.json            本次运行的完整配置
run_manifest.json               代码、输入、运行时和产物 SHA-256
*_failures.jsonl                逐实例结构化失败；无失败时为空文件
*_example_schedule.json         首个场景的完整调度轨迹
```

主指标使用失败惩罚后的 score：

```text
成功：score_ratio = masked_mlp_makespan / HEFT_makespan
失败：score_ratio = evaluation.failure_penalty_ratio（默认 10.0）
mean_ratio = mean(score_ratio)，失败仍保留在总样本分母中
```

数值越低越好，`1.0` 表示与 HEFT 持平。摘要还同时报告 `failure_count`、真实 `failure_rate`、`valid_schedule_rate` 和成功子集均值；失败明细写入 JSONL。完整字段、CLI 非零退出和 manifest 契约见 [P0-08 失败统计与运行清单](doc/P0-08失败统计与运行清单.md)。

## 运行测试

```powershell
python -m pytest
```

测试覆盖场景合法性、DAG 环检测、ready mask、HEFT 调度、神经策略训练/保存/加载，以及完整 pipeline 输出。`tests/fixtures/golden/heft_cases.json` 还固化了 10 个手工推导的 HEFT 黄金案例，覆盖插入式时间线、非对称通信、固定时延和确定性 tie-break。

正确性门禁还包括独立实现路径：10 个黄金案例在人工 fixture、生产 HEFT 和独立 oracle 之间逐字段比对；40 个固定 seed 的随机 DAG 分别运行 HEFT、CPOP、Greedy-EFT 和 Random，共检查 160 份调度；10 类故障注入验证遗漏、重复、越界、时间、依赖、资源重叠和 makespan 错误均会被拒绝。设计边界和复核方法见[独立验证器设计与测试报告](doc/独立验证器设计.md)。

`tests/fixtures/golden/optimal_cases.json` 进一步冻结了 10 个小图的 CPOP 完整决策轨迹、解析下界和精确最优 makespan。精确 solver 穷举所有 active schedule，只用于不超过 8 个任务的小图正确性测试；算法定义、证明边界和结果见[CPOP 与小图精确 Oracle 报告](doc/CPOP与精确Oracle设计.md)。

上述输入、环境、HEFT、CPOP、小图最优参考和跨平台证据已汇总到 [G1 正确性门禁评审包](doc/G1正确性门禁评审.md)，并由成员 A [独立复核通过](doc/G1独立复核记录.md)。P0-10 openEuler smoke 也已由 A 从远端不可变提交[独立复核通过](doc/P0-10独立复核记录.md)。这些结论只覆盖当前静态模型正确性和 openEuler 用户态兼容性，仍不能宣称当前学习策略优于 HEFT。

## 获取公开 STG benchmark

P1-B01 冻结了 Zenodo `10.5281/zenodo.18927122` 中 180 个 rnc50 hetero JSON 的来源字节、CC BY 4.0 署名、TriSched projection v1 和 120/30/30 split。原始数据不提交 Git，首次在线获取后可离线复核：

```powershell
python scripts/fetch_stg_benchmark.py
python scripts/fetch_stg_benchmark.py --offline
```

脚本先校验 125,500 字节 archive 的 SHA-256，再安全解包并逐个重算 source/Scenario hash；任何源文件、projection 或 split 变动都会失败。当前投影只保留 STG topology、duration 和 predecessor output data，不保留上游 GPU/core/memory capability，不能表述为完整 GrapheonRL 系统复现。许可证决策与固定 hash 见 [P1-B01 公开基准、许可证与冻结划分](doc/P1-B01公开基准与许可证.md)，远端 CI、10 实例抽查和故障注入见 [A 的独立复核记录](doc/P1-B01独立复核记录.md)。

## 训练公开 STG 行为克隆基线

获取数据后，使用固定的 120 个 train teacher 和 30 个 validation reference 训练纯 BC 基线：

```powershell
python scripts/fetch_stg_benchmark.py --offline
python -m trisched train-bc --config configs/stg_bc.json
```

命令生成 teacher/reference manifest、训练曲线、best/last checkpoint、逐实例 validation 诊断、空失败 JSONL 和可核验 run manifest。训练入口只加载 train/validation；test 被用途门禁禁止用于 teacher、梯度或 checkpoint 选择，本任务不会输出 test 指标。当前 BC 的 validation ratio 为 1.0，表示成功复制 HEFT，不代表性能领先。契约、结果和 hash 见 [P1-A01 HEFT teacher 与公开 STG 行为克隆基线](doc/P1-A01HEFT教师与BC基线.md)；B 已在完全不含 test 原始字节的隔离数据根上双次重跑，并[独立复核通过](doc/P1-A01独立复核记录.md)。

## 训练公开 STG masked PPO

P1-A02 主策略删除 `is_heft_task` 和 `is_heft_pair`，从 14 维 BC warm start 开始，用增量 makespan shaping、GAE 和 clipped PPO 训练 3 个 seed：

```powershell
python scripts/fetch_stg_benchmark.py --offline
python -m trisched train-ppo --config configs/stg_ppo.json
```

训练会在每个完整 PPO epoch 后原子保存 actor/value 参数、Adam、RNG、history 和 best 选择状态。异常退出后使用相同配置、代码、数据和输出目录恢复：

```powershell
python -m trisched train-ppo --config configs/stg_ppo.json --resume
```

如果首次运行指定了 `--output <dir>`，恢复时也必须指定同一目录。配置、代码、数据、teacher/reference、warm start 或状态 hash 任一变化都会以结构化错误拒绝恢复；普通 best/last 推理 checkpoint 不能冒充训练状态。恢复先复制到同盘隐藏 staging 目录，所有 seed、summary 和 manifest 成功后才替换正式输出；任一校验或训练异常都会清理 staging 并保持正式目录逐字节不变。run manifest 记录 `publication_mode=staging_directory_swap`。该修复已由 B [第二轮独立复核通过](doc/P1-03第二轮独立复核记录.md)。

正式 validation 的 3 个 best seed ratio 为 `0.807240 / 0.623254 / 0.739086`，均为 30/30 合法、零失败、零非法动作；其中 PPO 改善 2 个 seed，另 1 个按冻结规则回退 BC warm start。seed-level mean 为 `0.723193`，但 population std 为 `0.075948`，每个 seed 仍有劣于 HEFT 的实例且 P95 ratio 全部大于 1。B 已在物理删除 test JSON 和 archive 的数据根上复跑正式配置，并完成 checkpoint、manifest、数学分支与配置注入复核。公开 test 完全未访问，因此当前只能表述为“validation 开发门禁通过”，不能宣称稳定优于 HEFT。详见 [P1-A02 设计与验收契约](doc/P1-A02MaskedPPO设计与验收.md)、其[独立复核记录](doc/P1-A02独立复核记录.md)和 [P1-03 断点续训契约](doc/P1-03PPO断点续训设计与验收.md)。

P1-A04 使用显式 `seed_extension` 绑定上述三 seed 来源 manifest，只训练 `20260720/20260721`：

```powershell
python -m trisched train-ppo --config configs/stg_ppo_5seed.json
```

五个 best validation ratio 为 `0.807240 / 0.623254 / 0.739086 / 0.658398 / 0.717970`，mean/std 为 `0.709190/0.064109`，150 个调度零失败、零非法动作。旧三 seed 的 24 个必需 artifact 与来源逐字节一致；50 个目标 artifact 全部可按 manifest 重算；两个新 checkpoint 复评与记录值一致。正式 run 绑定干净提交 `a38c530...`。B 又从 detached `8692b61...` 在 120 train、30 validation、0 test 的隔离根正式预算复跑，核心 artifact 和归一化 summary 与 A 包一致。完整合同、hash 与限制见 [P1-A04 五种子扩展设计与正式结果](doc/P1-A04五种子扩展设计与正式结果.md)及其[独立复核记录](doc/P1-A04五种子扩展独立复核记录.md)。这仍不能证明 OOD 或 public test 领先。

## P1-A03 task-GNN（已关闭）

当前已实现 task-GNN 的前向、完整解析梯度、裁剪 Adam、合法动作接口、只读 frozen graph state 和无 pickle checkpoint。它继续使用相同 14 维 teacher-free 候选特征，仅从其中复用 workload、upward-rank、入度和出度构造任务节点，沿 DAG 分别做一次前驱/后继均值消息传递，再与原候选表示融合打分。该接口已有同 seed 确定性、图结构敏感性、动作 mask、中心有限差分、冻结重放、参数量和 checkpoint 往返测试。

task-GNN 已通过专用函数完成合成 4/2/0 数据的微型 BC/PPO：零失败、零非法动作，PPO 无改善时正确回退 BC warm start。完整 epoch 状态覆盖 actor/value、两套 Adam、双 RNG、history 和 best，微型连续/中断/恢复逐字段一致。独立 CLI 现已生成 resolved config、teacher/reference、BC warm start、actor/value best/last、training state、curve、逐实例诊断、summary 和可重算 run manifest：

```powershell
python -m trisched train-task-gnn --config configs/stg_task_gnn.json
python -m trisched train-task-gnn --config configs/stg_task_gnn.json --resume
```

恢复沿用同盘 staging 目录交换；微型测试已证明连续/恢复的 summary、曲线和全部模型/状态数组一致，后置 seed 写出失败时正式目录逐字节不变。成员 B 已从远端不可变提交独立复跑 32-artifact、15 对 NPZ、后置失败目录快照和 MLP 旧入口并[复核通过](doc/P1-A03独立复核记录.md)。正式参数量 GNN/MLP 为 `1008/512`；120 个 train 场景的 frozen state 峰值 RSS 增量约 53.42 MiB，graph 按 Scenario 共享。

正式 task-GNN 3-seed validation ratio 为 `0.754139 / 0.633578 / 0.683664`，mean `0.690460`，低于 MLP 的 `0.723193`；90 对逐实例为 `50/19/21`，场景跨 seed 均值为 `20/2/8`，全部合法且零非法动作。但分层配对 bootstrap 的 95% CI 为 `[-0.083063, 0.007113]`，仍跨 0，且三个 seed 的 P95 均大于 1。按预先停止规则，主模型继续保留 MLP，task-GNN 只记为方向性改善、未证实，不追加新变量补救。B 已重算 artifact、状态、配对 CI，并用六个 best checkpoint 复评 180 次 validation 调度，makespan 最大误差为 0。公开 test 仍未访问；完整证据见[正式对照报告](doc/P1-A03正式对照报告.md)和[正式结果独立复核](doc/P1-A03正式结果独立复核记录.md)。

## P1-B02 评测契约与自动报告（正式 development 候选待 A 复核）

机器可读契约 [`configs/p1_b02_evaluation_contract.json`](configs/p1_b02_evaluation_contract.json) 已冻结 masked MLP 的 5 seeds、HEFT 参考、全部 baseline/ablation seed 数、ID + 三类 OOD 切片、失败惩罚 `10.0`、scheduler-only wall-clock 时延、10,000 次 seed→scenario bootstrap 和一次性公开 test gate。报告器严格检查 `seed × scenario` 笛卡尔积、场景 ID/hash、HEFT=1、失败计分、evidence hash 和 gate receipt，并输出 JSON、策略切片 CSV、逐 seed CSV、主策略配对 CSV 和 artifact manifest。JSON 的 `seed_mean_ratios` 按 seed 稳定排序；manifest 对四份声明产物记录字节数与 SHA-256。

```powershell
python -m trisched build-report `
  --contract configs/p1_b02_evaluation_contract.json `
  --evidence <development-evidence.json> `
  --output <empty-output-directory>
```

先从冻结 validation split 确定性物化 30 个 ID 场景和三类各 30 个 OOD 场景：

```powershell
python -m trisched materialize-ood `
  --contract configs/p1_b02_evaluation_contract.json `
  --benchmark-manifest data/benchmarks/stg-rnc50-hetero-v1.json `
  --raw-root outputs/benchmarks/stg-rnc50-hetero-v1/raw `
  --output outputs/p1-b02-development-slices
```

命令只以 `validation/model_selection` capability 加载官方 split，并原子发布 `id_validation`、`ood_size`、`ood_ccr` 和 `ood_system`。冻结清单为 [`data/benchmarks/p1-b02-development-slices-v1.json`](data/benchmarks/p1-b02-development-slices-v1.json)，绑定契约 canonical hash、benchmark manifest SHA-256、每个源文件/Scenario hash、变换 provenance、每个场景文件 hash 和四个切片 aggregate hash；目标目录已存在时拒绝覆盖。只读 evidence producer 重新验证全部字节、来源和确定性变换，只计时 `runner.schedule(...)` 调用，并在停止计时后执行生产与独立 validator；HEFT 失败会阻断报告，其他策略失败保留惩罚行。

B 针对[首轮独立复核记录](doc/P1-B02独立复核记录.md)中的 R1–R3 输出全部逐 seed mean 和 `evaluation_per_seed.csv`，在 claim/final 两处执行冻结 receipt basename，并将授权时间冻结为严格的 `YYYY-MM-DDTHH:MM:SSZ`。A 已从提交 `071b861...` 独立构造不同 seed mean、重算四份 artifact、注入错误 basename/receipt 字段/六类非法 UTC 并[二次复核通过](doc/P1-B02第二轮独立复核记录.md)。OOD 路径的 120 个场景、8 类篡改、外层计时、失败语义和 producer/report hash 互操作也已通过 A 的[第三轮独立复核](doc/P1-B02OOD证据路径第三轮独立复核记录.md)。

B 已从干净提交 `d9c5c10...` 用正式入口运行 development 评测：

```powershell
python scripts/run_p1_b02_development.py
```

正式 evidence 含 2040 条完整记录，全部成功且 failure/illegal 为 0。masked MLP 的 ID/size/CCR/system mean ratio 分别为 `0.709190/1.568302/0.406325/0.965250`；size-OOD 明显劣于 HEFT，system-OOD 配对 CI 跨 0，因此报告正确给出 `release_publishable=false`。记录笛卡尔积、9/9 checkpoint hash 和 4/4 报告 artifact hash 已由 B 自审可重算；完整命令、数字和边界见[正式 Development 结果记录](doc/P1-B02正式Development结果记录.md)。P1-B02 仍部分完成，下一步由 A 从用户推送后的不可变提交独立复核；public test 继续禁止。

## openEuler CPU smoke

在已启动 Docker Linux engine 的 Windows 或 Linux 主机上，使用固定 digest 的 openEuler 24.03 LTS-SP4 镜像执行一次性 CPU smoke：

```powershell
$image = "openeuler/openeuler:24.03-lts-sp4@sha256:17c15554be2a5bc46023acb6e04d609d77642b8c20e236e88deb18e41ae4558e"
docker pull $image
.\scripts\run_openeuler_smoke.ps1
Get-Content outputs\p0-10-openeuler\validation.json
```

脚本会在全新虚拟环境中安装非 editable 包，执行完整测试、pipeline 和 checkpoint 复评，并把证据写到被 Git 忽略的 `outputs/p0-10-openeuler/`。容器使用 `--rm`，成功或失败退出后均不作为交付物保留。固定镜像、结果、已知失败和 WSL2 容器边界见 [P0-10 openEuler 验证记录](doc/P0-10openEuler验证记录.md)及其[独立复核记录](doc/P0-10独立复核记录.md)。

## 接入外部调度器

`evaluation.schedulers` 可混合 registry 中的内置 runner 与外部进程 adapter。外部程序每个 Scenario 从 stdin 接收一个版本化 JSON，并在 stdout 返回完整 schedule；结果与内置策略进入同一 evaluator、生产 validator 和独立 validator。

仓库提供 `examples/external_heft_scheduler.py` 作为协议参考程序；它应由 pipeline 按配置启动，不能脱离 JSON stdin 单独运行。完整配置、请求/响应字段、退出码 3 和稳定错误码见 [P0-06 外部调度器接口](doc/外部调度器接口.md)。参考 external HEFT 只验证进程边界，不算新的算法基线。

## 生成可查看的场景 JSON

```powershell
python -m trisched generate --config configs/smoke.json --output outputs/generated
```

## 校验场景输入

Scenario 的正式结构定义位于 [`schemas/scenario.schema.json`](schemas/scenario.schema.json)。除 JSON Schema 可表达的字段和类型约束外，运行时还检查 ID 连续性、通信矩阵维度、有限数值、边端点、重复边和 DAG 无环。

```powershell
python -m trisched validate-scenario --input outputs/generated/test/test-0000.json
```

合法输入输出场景规模和内容 hash，退出码为 0；非法输入输出带稳定 `code` 和 JSONPath 风格 `path` 的单行 JSON，退出码为 2。完整错误契约和 fixture 见[Scenario 输入契约](doc/Scenario输入契约.md)。

## 加载 checkpoint 复评

无需重新训练即可加载冻结模型，并在确定性重建的 validation 或 test split 上复评：

```powershell
python -m trisched evaluate `
  --config configs/smoke.json `
  --checkpoint outputs/smoke/masked_mlp.npz `
  --split test `
  --output outputs/checkpoint-eval
```

命令会生成 `evaluation_summary.json`、逐实例 CSV、失败 JSONL、示例调度和 `run_manifest.json`；摘要记录 checkpoint 的 SHA-256，manifest 记录配置、数据、代码、依赖及所有声明产物的 SHA-256。

## 目录

```text
configs/smoke.json       最小训练与评测配置
configs/stg_bc.json      公开 STG teacher/BC 冻结配置
configs/stg_ppo.json     公开 STG 3-seed masked PPO 配置
configs/stg_ppo_5seed.json 绑定旧证据、只补两 seed 的 5-seed 配置
configs/stg_task_gnn.json 公开 STG 3-seed task-GNN 单变量配置
configs/p1_b02_evaluation_contract.json ID/OOD、5-seed 与 test gate 契约
schemas/                 Scenario JSON Schema
trisched/scenario.py     场景 schema、校验、生成和 hash
trisched/env.py          调度环境、插入式时间线和生产合法性检查
trisched/oracle.py       独立 HEFT、upward rank 和合法性验证
trisched/exact.py        小图精确分支定界 solver 与解析下界
trisched/policies.py     统一策略接口与 HEFT/CPOP/Greedy/Random
trisched/schedulers.py   scheduler registry、外部进程 adapter 和稳定诊断
trisched/learning.py     可变特征 Masked MLP、HEFT 模仿和 legacy REINFORCE
trisched/gnn.py          14 维输入 task-GNN、DAG 消息传递和 checkpoint
trisched/bc.py           MLP/task-GNN 冻结 teacher、BC best/last 和防 test 泄漏流程
trisched/ppo.py          MLP/task-GNN 增量奖励、GAE/PPO、epoch 状态与 run 事务
trisched/evaluation.py   逐实例评测、统计与标准结果文件
trisched/reporting.py    P1-B02 evidence 校验、配对统计与自动报告
trisched/ood.py          P1-B02 development ID/OOD 物化与只读 evidence producer
trisched/cli.py          pipeline/train-bc/train-ppo/train-task-gnn 等命令
trisched/benchmark.py    公开 STG loader、冻结 split 与来源校验
data/benchmarks/         第三方来源/许可证元数据和冻结 manifest
examples/                外部 scheduler 协议参考程序
tests/                   单元与集成测试
```

## 当前边界

- 每次评测一个静态 DAG，资源和链路在单个 episode 内不变化；
- 目标函数仅为 makespan；
- 精确 solver 具有指数复杂度，仅用于不超过 8 个任务的小图，不参与常规训练或全量评测；
- task-GNN 已使用一层任务 DAG 消息传递，但资源关系仍只由手工候选特征表达；
- legacy `pipeline` 仍使用 REINFORCE；公开 STG 已有独立 masked PPO，task-GNN 已完成正式 validation 单变量对照但未通过稳健替换门禁，尚无课程学习；
- P1-B02 已冻结 ID/OOD、5-seed、自动报告口径和 development 场景 manifest；B 已生成正式 development 候选，但尚待 A 独立复核，public test 继续禁止；
- 正式 development 显示 MLP 在 ID/CCR-OOD 上有明确优势，在 system-OOD 上不确定，在 size-OOD 上退化到 `1.568302`；自动门禁不允许发布；
- task-GNN 的 epoch 与目录级断点续训、正式 artifact 和 checkpoint 复评均已由 B 从不可变提交复核；当前不支持 minibatch 内恢复或跨代码/配置迁移，原始文本 hash 还会受 LF/CRLF 检出策略影响，release 前须补规范化 hash 和跨 clone 测试。

## 开源许可证

本项目采用 [Apache License 2.0](LICENSE)。第三方代码、数据和模型仍须分别核验并记录其许可证。
