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
- 用 HEFT 行为克隆预训练两层 NumPy MLP，并将 HEFT 当前决策作为教师先验特征，再以 episodic REINFORCE 学习残差；
- 一条命令完成训练、验证、测试并产出 `summary.json`。

这是用于验证赛题接口、环境正确性和实验流程的 MVP，不是最终获奖模型。下一阶段应在保持接口不变的前提下，将候选特征 MLP 替换为任务图/资源图 GNN，并升级为 PPO。

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
*_example_schedule.json         首个场景的完整调度轨迹
```

主指标为：

```text
mean_ratio = mean(masked_mlp_makespan / HEFT_makespan)
```

数值越低越好，`1.0` 表示与 HEFT 持平。

## 运行测试

```powershell
python -m pytest
```

测试覆盖场景合法性、DAG 环检测、ready mask、HEFT 调度、神经策略训练/保存/加载，以及完整 pipeline 输出。`tests/fixtures/golden/heft_cases.json` 还固化了 10 个手工推导的 HEFT 黄金案例，覆盖插入式时间线、非对称通信、固定时延和确定性 tie-break。

正确性门禁还包括独立实现路径：10 个黄金案例在人工 fixture、生产 HEFT 和独立 oracle 之间逐字段比对；40 个固定 seed 的随机 DAG 分别运行 HEFT、CPOP、Greedy-EFT 和 Random，共检查 160 份调度；10 类故障注入验证遗漏、重复、越界、时间、依赖、资源重叠和 makespan 错误均会被拒绝。设计边界和复核方法见[独立验证器设计与测试报告](doc/独立验证器设计.md)。

`tests/fixtures/golden/optimal_cases.json` 进一步冻结了 10 个小图的 CPOP 完整决策轨迹、解析下界和精确最优 makespan。精确 solver 穷举所有 active schedule，只用于不超过 8 个任务的小图正确性测试；算法定义、证明边界和结果见[CPOP 与小图精确 Oracle 报告](doc/CPOP与精确Oracle设计.md)。

上述输入、环境、HEFT、CPOP、小图最优参考和跨平台证据已汇总到 [G1 正确性门禁评审包](doc/G1正确性门禁评审.md)，并由成员 A [独立复核通过](doc/G1独立复核记录.md)。P0-10 openEuler smoke 也已由 A 从远端不可变提交[独立复核通过](doc/P0-10独立复核记录.md)。这些结论只覆盖当前静态模型正确性和 openEuler 用户态兼容性，仍不能宣称当前学习策略优于 HEFT。

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

命令会生成 `evaluation_summary.json`、逐实例 CSV 和示例调度，其中摘要记录 checkpoint 的 SHA-256。

## 目录

```text
configs/smoke.json       最小训练与评测配置
schemas/                 Scenario JSON Schema
trisched/scenario.py     场景 schema、校验、生成和 hash
trisched/env.py          调度环境、插入式时间线和生产合法性检查
trisched/oracle.py       独立 HEFT、upward rank 和合法性验证
trisched/exact.py        小图精确分支定界 solver 与解析下界
trisched/policies.py     统一策略接口与 HEFT/CPOP/Greedy/Random
trisched/schedulers.py   scheduler registry、外部进程 adapter 和稳定诊断
trisched/learning.py     Masked MLP、HEFT 模仿和 REINFORCE
trisched/evaluation.py   逐实例评测、统计与标准结果文件
trisched/cli.py          pipeline/generate 命令
examples/                外部 scheduler 协议参考程序
tests/                   单元与集成测试
```

## 当前边界

- 每次评测一个静态 DAG，资源和链路在单个 episode 内不变化；
- 目标函数仅为 makespan；
- 精确 solver 具有指数复杂度，仅用于不超过 8 个任务的小图，不参与常规训练或全量评测；
- 学习策略使用手工候选特征，还未使用 GNN；
- REINFORCE 是最小训练闭环，尚未加入 PPO、课程学习、OOD 数据与多随机种子报告；
- 当前合成场景用于工程验证，正式实验需要接入 STG/GrapheonRL benchmark 和固定隐藏测试集。

## 开源许可证

本项目采用 [Apache License 2.0](LICENSE)。第三方代码、数据和模型仍须分别核验并记录其许可证。
