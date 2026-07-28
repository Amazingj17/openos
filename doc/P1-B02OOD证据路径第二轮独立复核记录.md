# P1-B02 OOD 证据路径第二轮独立复核记录

- 任务：`P1-B02-OOD1-REVIEW2`
- 复核人：成员 A
- 日期：2026-07-18
- 被复核提交：`1e3c5f7516541892621595fa5dabbcdd362bbf32`
- 结论：R1/R2 定点修复通过；生产 evidence 与自动报告的 ID hash 语义不兼容，整体退回

## 1. 基线、范围与隔离

用户确认推送后，本地主工作区 `HEAD` 与 `origin/main` 均指向被复核提交。GitHub 443 端口当时不可达，无法刷新远端 Actions；A 仍从该不可变 Git 对象建立干净 detached worktree `outputs/p1-b02-ood1-a-rereview-worktree`。复核脚本位于 Git 忽略目录，不导入 `tests/test_p1_b02_ood.py` 或 B 的 fixture/helper。

A 使用生产契约、冻结 development manifest 和已由首轮审计通过的 120 个场景。所有 runner 均为 A 新写的串行合法桩或故障桩：没有运行 HEFT、PPO、GNN 等真实策略，没有加载 checkpoint，没有读取 public test，也没有启动训练。

## 2. R1/R2 定点复核通过

### R1：非法调度证据

- adapter 直接抛 `scheduler_invalid_schedule`：结构化 code 保留，`illegal_action_count=1`；
- 真实 `PolicySchedulerRunner` 包装 A 的非法动作策略：返回 `(-101,-202)` 后稳定转换为 `scheduler_invalid_schedule`，details 保留 task/resource/reason，`illegal_action_count=1`；
- 生产 evidence 中 Random 的 600 条与 masked MLP 的 600 条记录均为 failure、penalty `10.0`、对应 code 和 illegal 计数；没有再降级为 `scheduler_execution_failed`。

### R2：错误返回类型

- 非参考 runner 返回 dict：没有泄漏 `AttributeError`，而是 `scheduler_invalid_response`、`actual_type=builtins.dict`、penalty `10.0`、illegal 0；task-GNN 三个 seed 共 360 条记录一致；
- HEFT 返回 dict：外层稳定报 `ood_reference_failed`，内层为 `scheduler_invalid_response`，未写出部分 evidence；
- fake scheduler 推进 4 ms、两个 validator 合计再推进 500 ms，producer 报告仍为 4 ms，说明返回类型检查和 validator 均未进入 scheduler-only 计时。

将下面第 3 节的 hash 差异仅在审计副本中诊断性规范化后，四个切片的自动报告均显示：Random 和 masked MLP 各 `failure_count=150`、`illegal_action_count=150`、`illegal_record_count=150`；task-GNN 为 failure 90、illegal 0。该副本只用于隔离验证 R1/R2，不是可发布证据。

## 3. 新阻塞 R3：生产 evidence 无法直接构建报告

未经修改的生产 evidence 直接传给 `build_evaluation_report()` 时失败：

```text
code = report_hash
path = $.slice_manifests.id_validation
message = slice manifest count/hash mismatch
```

根因是同一字段存在两种顺序语义：

- materializer 以 manifest entries 的源顺序计算 `scenario_set_sha256`，冻结 ID 值为 `fea76e27e4c0a90e1a3067b9367f7e22372c5463e23b65abaf70aa5695010b1b`；
- report validator 先按 `scenario_id` 排序再计算，实际期望 `2aee0e6749cc1a11ba44ee083d27ca7940cd5db5654eb8d7ab33f4fbe842635e`；
- producer 直接把前一个值复制到 evidence，因此报告必然拒绝真实 ID 切片；三个 OOD 切片的 ID 本来就是字典序，两个算法偶然得到相同值；
- B 的合成 fixture 也使用自然排序 ID，所以新增自动报告测试未暴露该冲突。

这不是性能失败，而是正式 evidence producer 与已冻结报告器之间的协议失败。真实策略即使全部运行成功，也无法从原始 evidence 生成 P1-B02 报告，因而不能放行 5-seed development/OOD。

## 4. 回归与可追溯证据

- P1-B02 OOD + reporting：`41 passed in 2.35s`；
- 全量：`229 passed in 12.94s`；
- Black check：4 个目标文件不需改动；
- `compileall`、detached worktree clean、提交区间 `git diff --check`：通过；
- 审计脚本 SHA-256：`b961c3904a532537ea5754a6fd1c30599a997518d0c1e0c598c0e395601f4e7e`；
- 行为报告 SHA-256：`c9514f324322ab79af0a32ed6dcbc8b5b57fc53e39afc73548d4fac148ca00a3`；
- 原始组合 evidence SHA-256：`0995fccfaa970028622c137f1fb717f06034f722e9ada4e5e954221443876429`；
- 诊断性规范化 evidence/report SHA-256：`cbbda05739b7a10af7380ab89a8372a91539d1ab07502659a9e631d263425671` / `344be0bdb83af6a406f56ea5574161bcd35e1f88049d9626014aa5c9aaf72ef4`。

冻结 manifest 仍为 86,559 bytes、SHA-256 `d9c6e22ee7991b5659a46b6503a98580e9f3262edc5381bc9a32b8cf44acf2be`，`test_accessed=false`、`public_test_materialized=false`。本轮未修改 materializer、manifest 或场景字节。

## 5. 结论与重新验收条件

A 确认 B 的 R1/R2 修复通过，但因 R3 使生产 evidence 无法直接进入自动报告，**退回 P1-B02-OOD1-REVIEW2**，P1-B02 保持“部分完成”。

下一步由 B 只修 evidence/report 的 hash 互操作：producer 应按报告器的稳定排序语义生成 evidence `slice_manifests`，同时继续通过 `producer.materialization_manifest_sha256` 绑定原冻结 manifest；不重新物化或修改已通过审计的 120 个场景。新增回归必须使用非字典序的源场景顺序，并断言未经人工修改的 producer evidence 可直接构建报告。用户推送后由 A 从新不可变提交复核；通过前不运行真实策略或 5-seed OOD，不访问 public test，不启动训练。

## 6. B 的 R3 修复候选（待 A 复核）

B 已按第 5 节重新验收条件完成候选：producer 复用报告器的稳定排序 set hash；原 manifest 继续通过独立文件 SHA-256 绑定。非字典序 fixture 证明 materialization/evidence 两类 hash 可同时保持各自语义，原始 evidence 无需人工改写即可构建报告。

B 另用冻结 120 场景和纯串行合法桩生成 2,040 条 evidence，ID hash 为 `2aee0e67...`，冻结 manifest 仍为 `d9c6e22e...`，报告直接构建成功；专项/兼容/全量分别为 41/26/229 项通过。以上属于修复方自测，不改变第 5 节退回结论；用户推送后仍须由 A 从新不可变提交独立复核。真实策略、5-seed OOD、public test 和训练均未运行。

后续状态：A 已从新不可变提交确认 R3 与 R1/R2 全部通过，OOD materializer/evidence producer 子阶段关闭。完整证据见 [P1-B02 OOD 证据路径第三轮独立复核记录](./P1-B02OOD证据路径第三轮独立复核记录.md)。
