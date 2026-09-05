# TriSched 混合数据集增强与 OOD 评测

## 1. 目标

混合数据流水线把“任务图来源”和“资源网络来源”独立建模，再装配成统一的
云—边—端 `Scenario`。原始数据永远只读，训练、验证和测试均使用字段完整的
增强场景，避免训练问题和验证问题定义不一致。

## 2. 数据分工

| 来源 | 在 TriSched 中的职责 | 保留内容 | 补充内容 |
| --- | --- | --- | --- |
| STG 18927122 | 真实约束任务图 | DAG、duration、data、cores、memory、CPU/GPU feature | 云边端资源在装配阶段独立抽取 |
| DAGBench | 多领域、多结构、多规模任务图；也可提供资源网络 | task、dependency、cost、edge size、原网络 | CPU 核、内存、GPU、任务类型；零 cost 标记映射为 `1e-6` |
| 合成生成器 | 覆盖极端约束与冒烟测试 | 全部由固定随机种子产生 | 内存统一由 GiB 转为 MiB |
| Topology Zoo | 真实网络连接结构 | `.graph` 的节点、边、带宽、时延 | 节点映射为 device/edge/cloud，并补 CPU、内存、GPU、能力标签 |

Topology Zoo 的流量矩阵和 pickle 样本不属于当前调度状态，且部分文件名在
Windows 上非法，所以下载器只获取固定提交中的 36 个可移植 `.graph` 骨架。

## 3. 场景装配

每个场景执行以下步骤：

1. 从指定任务池按权重配额抽取任务图；
2. 从指定网络池按权重配额抽取资源图；
3. 将内存统一为 MiB，通信数据统一按 MB、带宽按 MB/s、时延按秒解释；
4. 检查每个任务的 CPU、内存、GPU 和 feature 约束；
5. 若外部拓扑没有任何节点能够运行某类任务，只扩展一个最高能力、优先 cloud
   的节点，使环境至少存在一个合法动作，并在清单中声明该规则；
6. 根据任务类型与资源层级生成显式 task×resource 执行时间矩阵；
7. 保存增强场景及逐场景来源记录。

默认配置把外部网络最多投影为 12 个计算节点，以控制当前 NumPy 训练器的
task×resource 计算量。投影采用 device、edge、cloud 轮询保留，而不是只留下
云节点；节点间带宽和时延来自裁剪前已计算的有效路径。把
`dataset.max_resource_nodes` 设为 `null` 可运行完整拓扑，但训练时间会明显增加。

默认训练任务比例为 STG 40%、DAGBench 40%、合成 20%；资源网络比例为
合成 50%、Topology Zoo 30%、DAGBench 20%。小规模实验采用“带覆盖的随机
配额”：样本数足够时，每个权重大于零且有数据的来源至少出现一次，然后随机
打乱顺序。这避免 8 个训练场景因偶然抽样完全漏掉某个数据源。

## 4. 固定划分

| 切片 | 任务图 | 资源网络 | 用途 |
| --- | --- | --- | --- |
| train | 训练任务家族 | 训练网络家族 | BC 与 PPO 更新 |
| id_validation | 同家族不同实例 | 同类但不同实例 | 早停和 checkpoint 选择 |
| dag_ood | 完全保留的 DAGBench 家族 | ID 网络 | 测任务结构泛化 |
| network_ood | ID 任务 | 完全保留的 Topology Zoo 拓扑 | 测网络泛化 |
| joint_ood | 保留 DAG 家族 | 保留网络拓扑 | 测联合分布偏移 |

默认保留 `scientific`、`ml`、`uav`、`v2x` 作为 DAG-OOD 家族，保留
`Gambia`、`Oxford`、`Sago`、`Zamren` 作为 Network-OOD。STG 的官方 test
划分不会用于训练或 checkpoint 选择。

## 5. 可复现与防泄漏

`dataset_manifest.json` 记录：

- 三种外部数据的固定 Zenodo 记录或 Git commit；
- 每个原始输入文件的 SHA-256；
- 每个场景的任务来源、网络来源及家族；
- 增强随机种子和增强器版本；
- 增强后 `Scenario` 的内容哈希；
- 字段单位及兼容性修复规则。

清单中的仓库内文件统一使用相对于项目根目录的 `/` 分隔路径，例如
`outputs/datasets/dagbench/workflows/.../graph.json`，不会记录盘符、用户名或
个人目录。若用户显式从仓库外加载数据，清单使用
`external-sha256://<digest>/<filename>` 内容地址，而不泄露主机路径。

划分发生在原始 DAG 身份/家族与原始网络身份/家族层面，而不是“增强文件训练、
原始文件验证”。同一原始任务实例不能同时进入 train 和 ID validation；被保留
的 OOD 家族不能进入训练池。

## 6. 命令与产物

仓库克隆后直接使用已提交的`data/enhanced-v1`：

```powershell
python -m trisched materialize-mixed-dataset --config configs/complex_dual_graph.json
python -m trisched train-dual-graph --config configs/complex_dual_graph.json
```

重新生成全部增强文件时，先运行`.\scripts\fetch_mixed_datasets.ps1`，再运行
`python -m trisched export-enhanced-datasets --config configs/complex_dual_graph.json`。

独立物化默认写入 `outputs/mixed-dataset/`。训练入口在自己的输出目录生成：

```text
outputs/complex-dual-graph/
├── actor.npz
├── critic.npz
├── training_curve.json
├── validation_per_instance.csv
├── summary.json
├── dataset_manifest.json
└── dataset_cache/
    ├── train/
    ├── id_validation/
    ├── dag_ood/
    ├── network_ood/
    └── joint_ood/
```

`summary.json.evaluation_splits` 分别报告四个评测切片中各调度器相对 HEFT 的
均值、标准差、最小值和最大值。主指标仍是 ID validation 上的
`mean(RL_makespan / HEFT_makespan)`，OOD 指标用于说明泛化而不参与模型选择。
