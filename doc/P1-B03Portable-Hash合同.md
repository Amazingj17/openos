# P1-B03 Portable Hash 合同与迁移说明

- 任务：`P1-B03-PORTABLE-HASH`
- 主责：成员 B
- 复核：成员 A
- 日期：2026-07-18
- 状态：实现完成，等待从远端不可变提交独立复核

## 目标与非目标

本任务解决同一 Git 文本 blob 在 Windows 不同 worktree 中因 LF/CRLF 检出策略不同而产生不同原始 SHA-256 的问题。合同同时保留两种不同含义：

1. `raw_sha256` 证明一次执行实际读取的精确字节；
2. `normalized_lf_sha256` 表示跨 clone/worktree 可比较的文本身份。

本任务不改写 P1-B02 或更早的 evidence，不把二进制 checkpoint 做文本归一化，也不以规范化 hash 替代运行时原始字节证据。

## 冻结合同

### 文本文件

`trisched.hashing.portable_text_hashes(path)` 返回：

```json
{
  "bytes": 123,
  "raw_sha256": "...",
  "normalized_lf_sha256": "...",
  "encoding": "utf-8",
  "normalization": "CRLF/CR to LF; no other transformation"
}
```

规范化算法只做三步：严格按 UTF-8 解码、把 CRLF 和孤立 CR 转成 LF、按 UTF-8 重新编码。它不删除尾随空格、不补末尾换行、不做 Unicode NFC/NFD 归一化。非 UTF-8 输入必须失败，不能静默替换字符。

### JSON 对象

`canonical_json_sha256(value)` 使用 UTF-8、键排序、无多余空白、`ensure_ascii=false` 和 `allow_nan=false`。该 hash 绑定解析后的 JSON 值，不绑定缩进和行尾；需要证明配置文件实际执行字节时仍须同时记录 `portable_text_hashes`。

### Git 属性和二进制文件

`.gitattributes` 对 Python、JSON/JSONL、Markdown、YAML、TOML、PowerShell、shell、文本和 CSV 固定 `eol=lf`，并把 NPZ、ZIP、DOCX、PDF 和常见位图显式标记为 binary。二进制产物只使用原始字节 SHA-256，不允许文本归一化。

## Producer 迁移规则

`scripts/run_p1_b02_development.py` 保留历史兼容字段 `script.sha256`，它继续表示本次执行脚本的精确字节；新的 `script.portable_text` 并列记录 raw 与 normalized-LF hash。未来新增的文本源码证据应使用同一双 hash 结构。

迁移遵循以下规则：

- 既有 evidence 原样保留，不能回填或重算成“看起来一致”的值；
- 新 evidence 若保留旧 `sha256` 字段，该字段仍表示精确执行字节；
- 跨 worktree 比较使用 `normalized_lf_sha256`；运行溯源使用 `raw_sha256`；
- checkpoint、archive 和文档容器等二进制文件只比较原始 SHA-256。

## 复现与验证

```powershell
python -m pytest -q tests/test_portable_hashing.py
git check-attr text eol -- .gitattributes trisched/hashing.py configs/smoke.json
python -m pytest -q
python -m compileall -q trisched scripts tests
git diff --check
```

本地主责验证结果：

- LF、CRLF、CR 三份不同原始字节得到三个不同 raw hash 和一个相同 normalized-LF hash；
- 非 UTF-8 文本被拒绝；canonical JSON 不受键顺序和空白影响；
- `git check-attr` 对约定文本返回 `text=set, eol=lf`，对 NPZ 返回 `text=unset`；
- P1-B03 专项 5 项通过；合并 P1-A05 测试后的全量回归通过；
- 历史 evidence 文件未修改。

## 独立复核要求

成员 A 必须在用户推送后从新的远端不可变提交执行：

1. 建立至少一个独立 worktree/clone，确认受管文本被检出为 LF；
2. 人工构造同内容 LF/CRLF 文件，确认 normalized-LF hash 相同而 raw hash 不同；
3. 对一个已跟踪文本 blob 比较 Git blob 内容 SHA-256 与 normalized-LF SHA-256；
4. 核对 P1-B02 历史 evidence 的 Git blob 未被改写；
5. 运行专项与全量回归，并记录远端 commit、命令、结果和 hash。

复核通过前，任务状态只能写“实现完成、待复核”，不能写“双签关闭”。

## 已知限制与回退

- 合同假定源码和配置是 UTF-8；其他编码必须先以独立任务迁移。
- 文件内容除行尾外的任何变化都会改变 normalized-LF hash，这是预期行为。
- `.gitattributes` 只约束后续检出/提交；已有工作树可能需要重新 clone 才能观察到一致行尾，不能用破坏性命令覆盖用户改动。
- 若新 producer 无法同时保留原始与规范化身份，应回退为只记录原始字节并阻塞 release，而不是伪造 portable hash。
