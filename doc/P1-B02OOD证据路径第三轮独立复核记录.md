# P1-B02 OOD 证据路径第三轮独立复核记录

- 任务：`P1-B02-OOD1-REVIEW3`
- 复核人：成员 A
- 日期：2026-07-18
- 被复核提交：`566f09b18cf448c664145db828e78935dd2c1b52`
- 结论：通过；关闭 OOD materializer/evidence producer 子阶段

## 1. 基线、范围与隔离

用户确认推送后，主工作区 `HEAD` 与 `origin/main` 均精确指向被复核提交。A 从该 Git 对象建立干净 detached worktree `outputs/p1-b02-ood1-a-review3-worktree`。独立审计脚本位于 Git 忽略目录，没有导入 B 的测试 fixture/helper。

复核包含两条路径：A 自建非字典序 validation 源顺序和小型物化集，覆盖 hash 语义与 R1/R2；再读取已经首轮审计通过的冻结 120 个 development 场景，生成完整生产契约 evidence 并直接构建报告。所有 runner 均为 A 新写的纯串行合法桩或故障桩，没有运行 HEFT、PPO、GNN 等真实策略，没有加载 checkpoint，没有访问 public test，也没有启动训练。

## 2. 非字典序端到端路径通过

A 固定 validation 源顺序为：

```text
review-zeta, review-alpha, review-mu
```

三类 hash 独立重算结果：

- materialization entries 源顺序 hash：`49239a4428e567ee6d6042400db4187eadb68594d708219a2b36bb116b899f30`；
- evidence 按 `scenario_id` 稳定排序 hash：`7ecd5ded8c45d91040757c4af4bdc3993d5d2911157132b8215c7dbe7ebb139c`；
- materialization manifest 文件 SHA-256：`b57b77ae8999c5ac1b0881e01c7069f5e5792de597e586d9e9f4bd2cfaf6daaf`。

前两个值按设计不同；producer 输出第二个值，并在 `producer.materialization_manifest_sha256` 精确记录第三个值。未经任何人工修改的原始 evidence 直接通过 `build_evaluation_report()`，生成四个切片报告。

## 3. R1/R2 与计时回归通过

小型生产路径共生成 204 条 evidence：

- Random adapter 非法 60 条：全部 `scheduler_invalid_schedule`，illegal 总数 60；
- masked MLP 进程内非法 60 条：全部 `scheduler_invalid_schedule`，illegal 总数 60；
- task-GNN 错误返回类型 36 条：全部 `scheduler_invalid_response`，illegal 总数 0。

每个切片的 Random/MLP/task-GNN failure 分别为 15/15/9，illegal 分别为 15/15/0。HEFT 返回 dict 时仍为外层 `ood_reference_failed`、内层 `scheduler_invalid_response`，且不写部分文件。fake scheduler 推进 4 ms、两个 validator 合计再推进 500 ms，报告 runtime 仍为 4 ms。

## 4. 冻结 120 场景生产路径通过

A 逐字节确认生产 materialization manifest 与 tracked frozen manifest 相同，随后独立重算：

- ID materialization 源顺序 hash：`fea76e27e4c0a90e1a3067b9367f7e22372c5463e23b65abaf70aa5695010b1b`；
- ID evidence 排序 hash：`2aee0e6749cc1a11ba44ee083d27ca7940cd5db5654eb8d7ab33f4fbe842635e`；
- manifest 文件 SHA-256：`d9c6e22ee7991b5659a46b6503a98580e9f3262edc5381bc9a32b8cf44acf2be`。

四切片各 30 场景。纯串行合法桩按生产契约生成 2,040 条 success evidence；producer 输出排序 hash 并绑定上述 manifest 文件 hash，原始 evidence 未经改写直接生成四切片报告。evidence/report SHA-256 为 `b5c524ad738b1a7b14501f7754722a0721bc3902003faffac1300a87266beb95` / `34cd16afe638ad82b5e2c1037fe8a0a1a702d917e42cd5947fe027fd9f06bdfb`。

## 5. 回归与审计证据

- P1-B02 OOD + reporting：`41 passed in 2.30s`；
- 全量：`229 passed in 12.92s`；
- Black check：2 个目标文件不需改动；
- `compileall`、detached worktree clean、提交区间 `git diff --check`：通过；
- 审计脚本 SHA-256：`0089226684047134c30d210215a46d8e09c21368b8363546012a9dd910da1f6f`；
- 审计报告 SHA-256：`1f17c5e631eb9ae48ddffa78cecb7af4a306a30c8393cb87a71de772c41e0ef9`；
- 小型非字典序 evidence/report SHA-256：`f1ff554c38dfedc79ea0f938517e5156a26ea82b0418897a9988346331c9c202` / `ccd42c8290b8851fc49286a778f34c158ce77110fe45b995a8c7155b9bd9e458`。

输出继续声明 `test_accessed=false`、`training_started=false`、`public_test_loaded=false`。本轮没有重新物化或修改冻结生产场景。

## 6. 结论与下一步

A **通过 P1-B02-OOD1-REVIEW3**。此前通过的 materializer/source/hash/篡改/计时、R1/R2 失败语义以及本轮 R3 hash 互操作合并关闭 OOD materializer/evidence producer 子阶段。

P1-B02 总任务仍为“部分完成”：当前只有 3-seed masked MLP 正式 validation artifact，尚未满足契约要求的 5 seeds，也没有运行任何真实策略的 development/OOD 报告。下一主任务交给成员 A：先冻结 5-seed artifact 扩展方案，只补 `20260720/20260721` 两个缺失 seed 并保持既有配置、训练预算、选模和 test 禁令；artifact 由 B 独立复核后，才允许 B 运行正式 development/OOD。公开 test 继续禁止。
