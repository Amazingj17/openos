# TriSched MVP

赛题十五“基于深度强化学习的云-边-端异构计算资源管理调度方法”的最小可执行实现。

它已经形成一个完整闭环：

- 参数化生成任务 DAG、云/边/端资源与通信矩阵；
- 使用插入式时间线模拟执行与跨节点通信；
- 通过统一接口运行 HEFT、Greedy-EFT、Random 和学习策略；
- 对所有“就绪任务 × 资源”候选执行严格合法动作掩码；
- 用 HEFT 行为克隆预训练两层 NumPy MLP，并将 HEFT 当前决策作为教师先验特征，再以 episodic REINFORCE 学习残差；
- 一条命令完成训练、验证、测试并产出 `summary.json`。

这是用于验证赛题接口、环境正确性和实验流程的 MVP，不是最终获奖模型。下一阶段应在保持接口不变的前提下，将候选特征 MLP 替换为任务图/资源图 GNN，并升级为 PPO。

## 快速运行

要求 Python 3.10+。唯一运行依赖是 NumPy。

```powershell
python -m pip install -r requirements.txt
python -m trisched pipeline --config configs/smoke.json
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

测试覆盖场景合法性、DAG 环检测、ready mask、HEFT 调度、神经策略训练/保存/加载，以及完整 pipeline 输出。

## 生成可查看的场景 JSON

```powershell
python -m trisched generate --config configs/smoke.json --output outputs/generated
```

## 目录

```text
configs/smoke.json       最小训练与评测配置
trisched/scenario.py     场景 schema、校验、生成和 hash
trisched/env.py          调度环境、插入式时间线、独立合法性检查
trisched/policies.py     统一策略接口与 HEFT/Greedy/Random
trisched/learning.py     Masked MLP、HEFT 模仿和 REINFORCE
trisched/evaluation.py   逐实例评测、统计与标准结果文件
trisched/cli.py          pipeline/generate 命令
tests/                   单元与集成测试
```

## 当前边界

- 每次评测一个静态 DAG，资源和链路在单个 episode 内不变化；
- 目标函数仅为 makespan；
- 学习策略使用手工候选特征，还未使用 GNN；
- REINFORCE 是最小训练闭环，尚未加入 PPO、课程学习、OOD 数据与多随机种子报告；
- 当前合成场景用于工程验证，正式实验需要接入 STG/GrapheonRL benchmark 和固定隐藏测试集。
