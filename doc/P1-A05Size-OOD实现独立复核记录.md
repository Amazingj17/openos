# P1-A05 Size-OOD 实现独立复核记录

- 任务：`P1-A05-IMPLEMENT-REVIEW`
- 被复核人：成员 A
- 复核人：成员 B
- 日期：2026-07-18
- 结论：通过；receipt 已生成，尚未开始正式训练

## 1. 复核基线与批准源码

B 从 `origin/main=06d3b0cd59e7b7b65fb4b38a7882c24b2dd53bb3` 建立干净 detached worktree。P1-A05 受控训练源码首次完整落在 `2ab2c66f1317637d41adb4f28426ee3c726fbc5c`；独立执行：

```powershell
git diff --exit-code `
  2ab2c66f1317637d41adb4f28426ee3c726fbc5c `
  06d3b0cd59e7b7b65fb4b38a7882c24b2dd53bb3 `
  -- .gitattributes trisched configs/p1_a05_size_robustness.json `
     requirements.txt requirements-lock.txt pyproject.toml
```

结果为零差异。因此 receipt 将 `2ab2c66...` 记为 `approved_source_commit`；后续 `9831b80...`、`06d3b0c...` 只增加发布/文档证据，没有改变受控训练源码。这样既避免 receipt 自引用，也保证正式运行时可重算源码树门禁。

## 2. 隔离数据根

复核工作树只复制了下列批准输入：

| 类别 | 数量 | 结论 |
| --- | ---: | --- |
| benchmark train/validation raw | 120 / 30 | 与冻结 manifest 一致 |
| public-test raw | 0 | 物理不存在 |
| archive | 0 | 物理不存在 |
| development 控制文件/场景 | manifest 1 + 场景 120 | 只用于去重，不进入训练 |
| P1-A04 warm-start | summary 1 + checkpoint 5 | 原始/参数 hash 全部一致 |

随后独立执行：

```powershell
python -m trisched prepare-p1-a05 --config configs/p1_a05_size_robustness.json
python -m trisched dry-run-p1-a05 --config configs/p1_a05_size_robustness.json
```

prepared 根包含 120 个 train、30 个 validation、60 个 synthetic-100、5 个 warm-start 和控制文件。60 个 synthetic 场景的 ID 与内容 hash 均为 `60/60` 唯一；synthetic、benchmark train、benchmark validation、development 四类之间 ID/hash 交集总数均为 0，data-boundary 违规总数为 0。

## 3. Dry-run 与负向门禁

两轮 rollout 都精确得到 `90 episodes / 6000 transitions`，并同时记录：

```text
checkpoint_loaded=false
optimizer_created=false
training_started=false
```

P1-A05 专项 15 项覆盖 unknown/missing 字段、冻结配置漂移、91 episodes、6001 transitions、重复/未知场景、public-test 放行和 receipt/source 绑定。receipt 缺失时，真实训练入口返回 exit 2/`p1_a05_review_missing`，正式输出目录不存在；拒绝发生在 checkpoint loader、优化器和训练之前。

## 4. 复核哈希

| 对象 | SHA-256 |
| --- | --- |
| 生产配置 canonical JSON | `62afb332f73cd790f77c0b9ec0e228c712bb101ad10904ab47abe390a70d3d34` |
| 预注册 canonical JSON | `9ab60842d87e6d906ab18ad135cda2a372be569ad39b28ce6f17c60f1ff64647` |
| prepared manifest | `3fe388b38325d575198f913c83ed58e63f52389d632eb23dde5de96cde1a96fa` |
| rollout dry-run | `b81fde389e73a45449f8caf190af23bd15cd43387b56f2dc0bf8f02551887b2a` |

五个 warm-start 的原始 SHA-256 和参数 SHA-256 逐项等于生产配置，不存在重序、替换或重新训练。

## 5. 回归与结论

```text
P1-A05 专项：15 passed
全量回归：260 passed in 25.50s
受控源码 diff：0
public test：forbidden / bytes absent
```

B 批准当前实现，并生成 [`configs/p1_a05_implementation_review.json`](../configs/p1_a05_implementation_review.json)。只有包含该 receipt 的提交被用户推送、远端 commit 可独立确认且工作树干净后，才能执行唯一一次正式 P1-A05 训练。receipt 入库不等于训练完成，也不解除 G3 或 public-test 门禁。
