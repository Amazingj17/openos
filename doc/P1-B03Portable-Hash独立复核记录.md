# P1-B03 Portable Hash 独立复核记录

- 任务：`P1-B03-PORTABLE-HASH-REVIEW`
- 被复核人：成员 B
- 复核人：成员 A
- 日期：2026-07-18
- 结论：通过；P1-B03 双签关闭

## 1. 独立环境

A 从 `origin/main=06d3b0cd59e7b7b65fb4b38a7882c24b2dd53bb3` 建立独立 detached worktree。全量回归为 `260 passed in 25.51s`，portable/release/docs 专项 9 项全部通过。`.gitattributes` 对源码、JSON 和 Markdown 返回 `text=set,eol=lf`，对 checkpoint/archive/docx 等二进制返回 `text=unset`。

## 2. Git blob 与行尾注入

对受跟踪 producer `scripts/run_p1_b02_development.py`，Git blob 内容 SHA-256 为：

```text
165bc8e304138bdd7f5735b5953716300b7f124c881685215fb97be3a4353447
```

独立 worktree 文件的 normalized-LF SHA-256 与该 Git blob 完全一致。另以相同文本人工构造 LF/CRLF 输入：

| 输入 | raw SHA-256 | normalized-LF SHA-256 |
| --- | --- | --- |
| LF | `e49c81e2...d0d78ee` | `e49c81e2...d0d78ee` |
| CRLF | `98ab4d3a...c2b6fc80` | `e49c81e2...d0d78ee` |

结果证明 raw hash 继续绑定实际读取字节，而 normalized-LF 只消除 CRLF/CR 差异；实现没有折叠其他内容差异。

## 3. 历史证据与 producer 迁移

历史 `outputs/p1-b02-development-evidence/development-evidence.json` 的 SHA-256 仍为：

```text
40d781cad7dff39553ba0c027b686e5710d19ceed62fa3bf7f0471a133feb744
```

旧 evidence 保留历史 `script.sha256`，没有回填或伪造 `portable_text` 字段。新 producer 同时写 raw 和 normalized-LF 身份，二进制 checkpoint 仍只使用 raw SHA-256，符合迁移合同。

## 4. 发布路径交叉验证

从远端提交连续构建两次 source bundle；两份均含 129 个 Git 文件，逐字节完全一致：

```text
00d887c4545687113fdb4998d50f5471d2b05e3f404604b62e748c5921bccce7
```

内部 manifest 的 bytes/SHA-256/Git blob 可全部重算，说明 portable 文本合同与直接读取 Git blob 的发布路径兼容。

## 5. 结论

P1-B03 的双 hash 语义、LF 属性、二进制边界、历史 evidence 不改写和确定性发布均通过独立复核。跨 worktree 比较使用 normalized-LF，单次运行溯源继续使用 raw；两者不得互相替代。
