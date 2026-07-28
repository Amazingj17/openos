# P1-B02 正式 Development ID/OOD 结果记录

- 任务：`P1-B02-DEVELOPMENT`
- 主责：成员 B
- 复核：成员 A
- 日期：2026-07-18
- 状态：B 正式候选已由 A 从远端不可变提交独立复跑并复核通过；P1-B02 development/OOD 评测子阶段关闭
- 边界：public test 未加载、未声明、未运行；本记录不是最终 test 结论

## 1. 结论

B 从干净提交 `d9c5c10d50ed970694fe7fb80f0e1ebea6be722b` 启动了一次冻结的 development 评测。17 个策略/seed 组合覆盖 4 个切片、每切片 30 个场景，得到严格完整的 `17 × 4 × 30 = 2040` 条记录。2040 条全部调度成功，failure、timeout、penalty 和 illegal action 均为 0。

冻结主策略 masked MLP 在 ID 和 CCR-OOD 上稳健优于 HEFT，在 system-OOD 上点估计略优但 95% CI 跨 0，在 size-OOD 上则明显退化到 `1.568302`。因此自动报告的门禁结果是：

```text
primary_zero_failures_and_illegal_actions = true
primary_mean_ratio_below_reference_on_every_reported_slice = false
release_publishable = false
```

当前只能声明“ID/CCR-OOD 有利、system-OOD 不确定、size-OOD 失败”，不能声明整体 OOD 泛化通过，不能进入最终发布或访问 public test。A 已确认证据可信并关闭 P1-B02 development/OOD 评测子阶段，但性能门禁仍失败，G3 保持阻塞。

## 2. 冻结输入与生产入口

| 项目 | 冻结值 |
| --- | --- |
| 评测契约 | `configs/p1_b02_evaluation_contract.json` |
| 契约 canonical SHA-256 | `a46986f5695d59612ad9bf2c3379393620418b3f324a1003b5b5772b110c639e` |
| development 场景根 | `outputs/p1-b02-development-slices-v2/` |
| 场景 manifest SHA-256 | `d9c6e22ee7991b5659a46b6503a98580e9f3262edc5381bc9a32b8cf44acf2be` |
| 生产入口 | `scripts/run_p1_b02_development.py` |
| 入口 SHA-256 | `21043791e333baa7390eed850f7a9b4c466b012e003cac535af21aaf41e1df26` |
| 评测提交 | `d9c5c10d50ed970694fe7fb80f0e1ebea6be722b` |
| 策略/seed 组合 | 17：HEFT 1、Greedy-EFT 1、CPOP 1、Random 5、BC 1、masked MLP 5、task-GNN 3 |
| 学习 checkpoint | 9：BC 1、masked MLP 5、task-GNN 3 |

生产入口会校验每个 checkpoint 的内部 seed 和特征 schema，并在 evidence 中记录相对路径、字节数、容器 SHA-256 和独立参数 SHA-256。形式评测要求 Git 工作树干净；本轮 evidence 记录 `working_tree_dirty=false`。

## 3. 正式命令与产物

```powershell
python scripts/run_p1_b02_development.py

python -m trisched build-report `
  --contract configs/p1_b02_evaluation_contract.json `
  --evidence outputs/p1-b02-development-evidence/development-evidence.json `
  --output outputs/p1-b02-development-report
```

| 产物 | SHA-256 |
| --- | --- |
| `development-evidence.json` 文件 | `40d781cad7dff39553ba0c027b686e5710d19ceed62fa3bf7f0471a133feb744` |
| `records` canonical JSON | `0cf814874381d2ad392b3ae193fe11726949eefed72500a1f45433953fa8f8dd` |
| `evaluation_report_manifest.json` 文件 | `dd38444a2ac326277b0ccad705529579a9422f59af2a4fab548f08245c32206e` |

`outputs/` 受 `.gitignore` 管理，不把本机正式产物伪装成 Git 源文件。A 复核时应从不可变提交和已冻结 checkpoint 独立重生成，再比对规范化记录与报告结果。

## 4. 正式结果

表中的“MLP 95% CI”是 seed→scenario 分层 bootstrap 区间；“MLP−HEFT 95% CI”负数表示 MLP 更好。时延只覆盖 `runner.schedule(...)` wall-clock，生产与独立 validator 不计入。

| 切片 | HEFT | Greedy | CPOP | Random | BC | masked MLP | task-GNN | MLP 95% CI | MLP−HEFT 95% CI | MLP P50/P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| `id_validation` | 1.000000 | 1.150492 | 1.068423 | 6.685566 | 1.000000 | 0.709190 | 0.690460 | [0.602979, 0.820387] | [-0.397021, -0.179613] | 144.093 / 1803.591 |
| `ood_size` | 1.000000 | 1.117451 | 1.074161 | 2.909796 | 1.000000 | 1.568302 | 1.573522 | [1.501942, 1.619980] | [0.501942, 0.619980] | 1423.321 / 1617.656 |
| `ood_ccr` | 1.000000 | 0.939135 | 1.044342 | 12.963540 | 1.000000 | 0.406325 | 0.389563 | [0.262441, 0.582477] | [-0.737559, -0.417523] | 136.614 / 1798.723 |
| `ood_system` | 1.000000 | 1.443704 | 1.449998 | 13.424820 | 1.000000 | 0.965250 | 0.950678 | [0.771467, 1.197684] | [-0.228533, 0.197684] | 147.240 / 1793.819 |

masked MLP 全部 seed mean 和 population std 为：

| 切片 | `20260717` | `20260718` | `20260719` | `20260720` | `20260721` | std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `id_validation` | 0.807240 | 0.623254 | 0.739086 | 0.658398 | 0.717970 | 0.064109 |
| `ood_size` | 1.601640 | 1.462686 | 1.610442 | 1.556301 | 1.610442 | 0.056501 |
| `ood_ccr` | 0.496232 | 0.281611 | 0.397921 | 0.484359 | 0.371503 | 0.078752 |
| `ood_system` | 1.111695 | 1.211152 | 0.869839 | 0.826791 | 0.806773 | 0.164501 |

## 5. 独立可重算自审

B 在报告生成后用与生产命令分离的审计脚本重算：

- `2040/2040` 策略×seed×切片×场景笛卡尔积完整，无重复键；
- 2040 条 `status=success`，`penalty_applied=false`，illegal action 总数 0；
- `records_sha256` 和契约 canonical SHA-256 可重算；
- `test_accessed=false`、`training_started=false`、`public_test_loaded=false`；
- 9/9 checkpoint 的文件大小、文件 hash、参数 hash、内部 seed 和特征 schema 可重算；
- 生产入口与场景 manifest hash 可重算；
- 报告 manifest 声明的 4/4 artifact 的字节数和 SHA-256 可重算；
- 全量回归 232 项通过，`compileall`、`git diff --check` 通过。

首次启动时，外层 PowerShell 调用被 5 秒执行器超时结束，但唯一的 Python 评测子进程 PID `19920` 未中断，从 11:53:39 持续运行至 12:08:40 并写出唯一 evidence。B 发现子进程存活后未启动第二轮，因此不存在根据结果挑选重跑。

## 6. 结果解释与风险

1. ID 结果复现了 P1-A04 的五 seed validation 数字，证明新的正式 runner 与已双签 checkpoint 互操作一致。
2. size-OOD 将场景扩展到 100 tasks 后，五个 MLP seed 全部劣于 HEFT，且区间与 1.0 完全分离。这是当前最明确的获奖风险。
3. system-OOD 点估计为 `0.965250`，但 seed std 为 `0.164501`，配对 CI 跨 0；不得用“平均小于 1”替代稳健性结论。
4. task-GNN 在 ID、CCR 和 system 上的点估计均略优于 MLP，但在 size 上同样失败。本结果不解除 P1-A03 的停损决策，不追加未预注册调参。
5. 时延是当前 Windows/Anaconda 开发机上的 scheduler-only wall-clock，不是 openEuler 原生资源或最终部署性能结论。

## 7. 下一步和不可越过的边界

1. A 已从 `origin/main=fab540c...` 的干净 detached worktree 独立装载 9 个 checkpoint，重生成 2040 条 evidence/report，并[复核通过](./P1-B02正式Development独立复核记录.md)。
2. B/A 两轮去除 wall-clock 后的 2040 条调度行为和非时延报告完全一致；size-OOD 原始失败结果已保留。
3. A 下一步开始 `P1-A05-SIZE-ROBUSTNESS-DESIGN`，只做根因分析与单一改进候选预注册，不对当前 30 个 size-OOD 场景反复调参。
4. B 并行负责 `P1-B03-PORTABLE-HASH`，固定跨 worktree 行尾和源码 hash 合同；该问题不改写本次原始 evidence。
5. public test 仍受一次性 gate 禁止；不运行 `claim-test-gate`，不接入 public-test loader。

## 8. A 独立复核更新

A 的唯一一次复跑用时 `896.5s`，生成的 evidence 文件 SHA-256 为 `e2822774...e48d9`，`records` SHA-256 为 `bca214d0...1f02f`。A 重算 9/9 checkpoint、4/4 报告 artifact、逐 seed mean/std 和 gate 均通过；全量回归 `232 passed in 13.29s`。

复核另发现非阻断 F1：同一 Git blob 在 B 主工作区为 LF、A detached worktree 为 CRLF，原始 runner SHA-256 因此为 `21043791...` / `860da9e3...`；统一归一化为 LF 后与 Git blob 完全一致。两份 evidence 都正确绑定各自实际字节，该差异不否定调度结果，但必须在 release 前修复。完整证据见 [P1-B02 正式 Development 独立复核记录](./P1-B02正式Development独立复核记录.md)。
