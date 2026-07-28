# P1-B02 正式 Development ID/OOD 独立复核记录

- 任务：`P1-B02-DEVELOPMENT-REVIEW`
- 原主责：成员 B
- 独立复核：成员 A
- 日期：2026-07-18
- 被复核提交：`fab540c94207a7d1a121abb5032bbc02744ea2fb`
- 结论：通过 P1-B02 正式 development/OOD 证据复核；关闭评测子阶段
- 性能门禁：不通过；`release_publishable=false`，G3 不启动

## 1. 独立环境与隔离边界

用户确认推送后，A 执行 `git fetch --prune origin`，确认：

```text
HEAD = origin/main = fab540c94207a7d1a121abb5032bbc02744ea2fb
```

A 从该远端 Git 对象建立干净 detached worktree：

```powershell
git worktree add --detach `
  outputs/p1-b02-development-a-review-worktree `
  origin/main
```

复核 worktree 只导入：

- `p1-b02-development-slices-v2`：120 个 development 场景和 1 个 manifest；
- BC checkpoint 1 个；
- masked MLP checkpoint 5 个；
- task-GNN checkpoint 3 个。

物化 manifest 独立检查为 `test_accessed=false`、`public_test_materialized=false`，4 个切片均为 30 场景。复核输出根共 138 个文件，不存在 public-test 命名文件，没有导入 public-test 原始 JSON 或 archive。整个复核期间未运行 `claim-test-gate`，未启动训练。

## 2. 代码、runner 和隔离回归

A 不导入 B 的测试 helper，从机器契约装载真实 checkpoint 后确认：

- runner grid 精确为 17 个策略/seed 组合；
- 9/9 学习 checkpoint 的路径、字节数、文件 hash、内部 seed 和特征 schema 与 B 候选一致；
- 正式入口静态扫描不存在 `claim-test-gate`、public-test loader 或 BC/PPO/task-GNN 训练调用；
- 从 `55a3a7a...fab540c` 的提交区间 `git diff --check` 通过；
- detached 全量回归 `232 passed in 13.29s`；
- 复核前后 detached 与主工作树均干净，忽略目录中的审计输出不改写 Git 源文件。

## 3. 唯一一次正式复跑

A 在输出目录不存在、Git 状态干净时执行：

```powershell
python scripts/run_p1_b02_development.py
```

命令持续 `896.5s`，全程保持同一外层连接，退出码为 0，只生成一份 evidence。A 没有启动第二轮、更换 seed 或修改结果。原始 evidence 未经改写即被冻结报告器接受，报告生成命令用时 `13.2s`。

复跑得到：

```text
17 policy/seed × 4 slices × 30 scenarios = 2040 records
status=success              2040
penalty_applied=false       2040
failure/timeout/illegal        0
test_accessed              false
training_started           false
public_test_loaded         false
```

HEFT 的 120 条记录全部 `ratio=score_ratio=1.0`。策略×seed×切片×场景笛卡尔积为 `2040/2040`，无重复键；每个 slice/scenario 在全部策略间的 scenario hash 一致。

## 4. 独立统计与 A/B 归一化对比

A 在 Git 忽略目录新写审计脚本，没有调用 B 的测试 helper。脚本独立实现 canonical JSON/file/checkpoint parameter SHA-256，重算笛卡尔积、逐 seed mean/std、门禁和报告 artifact。

由于 `runtime_ms` 是 wall-clock，A/B 两轮的 raw evidence 和 `records_sha256` 必然不同。A 只去除每行 `runtime_ms`，按 policy/seed/slice/scenario 稳定排序后逐字段比较：

- A/B 2040 条归一化调度记录完全一致；
- 报告只去除 runtime、evidence file/records hash、commit 和原始行尾 script hash 后，所有非时延聚合、bootstrap、配对比较和 gate 完全一致；
- 9/9 checkpoint 的容器与独立参数 hash 可重算；
- 4/4 报告 artifact 的字节数和 SHA-256 可重算。

masked MLP 的独立重算值为：

| 切片 | mean ratio | seed std | MLP−HEFT 配对 95% CI | 结论 |
| --- | ---: | ---: | --- | --- |
| `id_validation` | 0.709190 | 0.064109 | [-0.397021, -0.179613] | 优于 HEFT |
| `ood_size` | 1.568302 | 0.056501 | [0.501942, 0.619980] | 明显劣于 HEFT |
| `ood_ccr` | 0.406325 | 0.078752 | [-0.737559, -0.417523] | 优于 HEFT |
| `ood_system` | 0.965250 | 0.164501 | [-0.228533, 0.197684] | 不确定，CI 跨 0 |

size-OOD 的五个 seed mean 为：

```text
1.601640 / 1.462686 / 1.610442 / 1.556301 / 1.610442
```

五个值全部大于 1.0，确认退化不是单 seed 偶然现象。报告 gate 独立重算为：

```text
primary_zero_failures_and_illegal_actions = true
primary_mean_ratio_below_reference_on_every_reported_slice = false
release_publishable = false
```

## 5. A 复跑产物 hash

| 证据 | SHA-256 |
| --- | --- |
| A evidence 文件 | `e2822774c9f692db5b51187716e46579e4a10553813211757669bacbda8e48d9` |
| A `records` canonical JSON | `bca214d0fa8f9af1e7d20c1ad4f1db04bb3aea973138d53974a292f48511f02f` |
| A report manifest 文件 | `647a1d615d4c2e96d72fbaf2e6ba2e315b8815a406a555a848cfe768ca3c1a7b` |
| A 审计脚本 | `77d7600fbae190b34cf554052c6ffa5793a63cb6647725509e10194a7882009c` |
| `evaluation_report.json` | `1e690ca60169f9e58da06b9d4d73da2dd1de62c71ffe599e18ef46bd99a2b31a` |
| `evaluation_per_slice.csv` | `cb7610e54c9169c7c72bba831c2466fb98f87bb7f2fe31782220aad863fac3cd` |
| `evaluation_per_seed.csv` | `438279c4882bd90806e06e9699158281dca0f7e4247610d33ad2df91515b7667` |
| `evaluation_primary_comparisons.csv` | `7d39d61d7ec1c492aaa1cb6b78c807e5ad81526ac409d2ecccf464a7fdc6d650` |

场景 materialization manifest 仍为 `d9c6e22e...acf2be`。A/B evidence 文件与 report artifact 因 wall-clock、commit 和工作树文本字节不同而不应逐字节相等；上节已对确定性行为和统计做精确对比。

## 6. F1：跨 worktree 原始脚本 hash 受行尾影响

A 发现同一 Git blob 在两个 Windows worktree 的原始文件 hash 不同：

| 对象 | SHA-256 |
| --- | --- |
| B 主工作区 LF 原始字节 | `21043791e333baa7390eed850f7a9b4c466b012e003cac535af21aaf41e1df26` |
| A detached worktree CRLF 原始字节 | `860da9e30108bc6cec313377ccd8cc3fa5365018598e0351451ddb1a33fcbda4` |
| A/B 统一归一化为 LF 及 Git blob | `21043791e333baa7390eed850f7a9b4c466b012e003cac535af21aaf41e1df26` |

三方归一化字节完全一致，两份 evidence 也都正确绑定各自实际执行的文件，因此 F1 不否定本次调度结果。但它会阻碍跨 clone/worktree 逐字节证据比较，必须在 release 前由 B 以新任务 `P1-B03-PORTABLE-HASH` 固定 `.gitattributes`/规范化源码 hash 合同并交 A 复核。

## 7. 结论与下一步

A **通过 `P1-B02-DEVELOPMENT-REVIEW`**。B 的正式候选和 A 的独立复跑在所有非 wall-clock 行为、统计和门禁上完全一致。关闭 P1-B02 development/OOD 评测子阶段；一次性 public-test 执行仍属于后续 release gate，本轮不启动。

“证据复核通过”不等于“模型性能通过”。size-OOD 稳定失败，system-OOD 区间跨 0，所以 G3 继续阻塞，不得访问 public test。用户推送本复核记录后：

1. A 开始 `P1-A05-SIZE-ROBUSTNESS-DESIGN`，只做 size 泛化根因分析与单一候选方案预注册，不立即反复调参；
2. B 开始 `P1-B03-PORTABLE-HASH`，解决 F1 并保持既有 evidence 不变；
3. 两个新任务均需对方独立复核；在新的预注册模型候选通过 development 门禁前，G3 和 public test 保持禁止。
