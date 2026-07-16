# P0-03 Scenario JSON Schema 与输入诊断设计

- 日期：2026-07-16
- 主责：成员 B
- 复核：成员 A
- 状态：已完成（成员 A 独立复核通过）

## 1. 目标

本任务把 Scenario 输入从“能被 dataclass 构造即可”升级为可发布、可诊断的两层契约：

1. `schemas/scenario.schema.json` 使用 JSON Schema Draft 2020-12 描述结构、必填字段、基本类型、取值范围和封闭对象；
2. `Scenario.from_dict/load/validate` 负责 JSON Schema 无法表达或不适合表达的跨字段不变量，并统一返回带错误代码和路径的诊断。

这项工作只收口静态 Scenario 输入，不改变调度、训练或评测数学语义，也不为 pipeline 配置定义 schema。

## 2. Schema 契约

顶层必填字段为：

```text
id, tasks, resources, edges, bandwidth, latency
```

`seed` 可省略，运行时默认值为 0。顶层、Task、Resource 和 Edge 均设置 `additionalProperties: false`，从而拒绝拼写错误或未版本化扩展。主要约束如下：

| 对象 | Schema 约束 |
| --- | --- |
| Scenario | 非空 `id`；tasks/resources 至少一项；edges 可为空 |
| Task | 非负整数 ID；workload 为正数 |
| Resource | 非负整数 ID；name 非空；kind 仅为 device/edge/cloud；speed 为正数 |
| Edge | source/target 为非负整数；data 非负 |
| bandwidth | 非空二维数组；每项为正数 |
| latency | 非空二维数组；每项为非负数 |

Draft 2020-12 schema 已用 `jsonschema 4.19.2` 的 `Draft202012Validator.check_schema` 做一次独立检查，10 个现有黄金场景全部通过。`jsonschema` 不是项目运行依赖；正式 CI 依靠固定 fixture 和内置运行时校验，因此不会为加载 Scenario 引入额外包。

## 3. 运行时不变量

以下约束需要比较多个字段或执行图算法，由 `Scenario.validate` 检查：

- task/resource ID 必须从 0 开始连续并与数组位置一致；
- bandwidth/latency 必须是 `resource_count × resource_count` 方阵；
- 所有浮点值必须有限，明确拒绝 NaN 和正负 Infinity；
- 边端点必须落在 task 范围内，不得自环或重复；
- 任务图必须无环；
- 直接调用 dataclass 构造函数时仍检查 ID、数值类型、资源名称/kind 和矩阵元素。

Python 标准 JSON parser 会接受非标准 `NaN/Infinity` 常量，因此 `Scenario.load` 使用 `parse_constant` 主动拒绝这些值。`1e309` 这类解析后溢出为 Infinity 的数字会在字段路径处由有限性检查拒绝。读取使用 `utf-8-sig`，兼容有无 UTF-8 BOM 的文件；其他编码返回 `encoding_error`。`Scenario.save` 设置 `allow_nan=false`，确保不会生成非标准 JSON。

## 4. 稳定错误格式

所有 Scenario 输入或不变量错误统一抛出 `ScenarioValidationError`，它仍是 `ValueError` 子类，保持现有调用方兼容。异常提供：

| 字段 | 含义 | 示例 |
| --- | --- | --- |
| `code` | 稳定的机器错误类别 | `matrix_shape` |
| `path` | JSONPath 风格字段位置 | `$.bandwidth` |
| `detail` | 面向人的具体说明 | `bandwidth must be ...` |
| `to_dict()` | 可直接序列化的三字段对象 | `code/path/message` |

冻结的错误代码包括：

| code | 使用场景 |
| --- | --- |
| `encoding_error` | 文件不是 UTF-8/UTF-8 with BOM |
| `json_syntax` | JSON 语法、行列位置错误 |
| `missing_field` | 缺少必填字段 |
| `unknown_field` | 出现 schema 未声明字段 |
| `type_error` | 字符串、整数、数值、数组或对象类型错误 |
| `non_finite` | NaN、Infinity 或数值溢出 |
| `value_error` | 正负范围、空值、端点、自环或枚举错误 |
| `id_sequence` | task/resource ID 不连续 |
| `matrix_shape` | 通信矩阵维度与资源数不一致 |
| `duplicate_edge` | 相同 source/target 重复 |
| `cycle` | 依赖图有环 |

字符串异常格式为 `<code> at <path>: <detail>`。调用方应按 `code/path` 判断类别，不应解析自然语言消息。

## 5. CLI

新增命令：

```powershell
python -m trisched validate-scenario --input path/to/scenario.json
```

合法输入返回退出码 0，并输出单行 JSON：

```json
{"valid": true, "scenario_id": "scenario-1", "task_count": 3, "resource_count": 2, "content_hash": "..."}
```

非法输入返回退出码 2：

```json
{"valid": false, "error": {"code": "cycle", "path": "$.edges", "message": "task graph must be acyclic"}}
```

文件不存在或无读取权限仍由操作系统 I/O 异常报告，不伪装成 Scenario 内容错误。

## 6. 固定非法 fixture 与测试

`tests/fixtures/invalid/scenario_cases.json` 固定 12 类非法 payload 及预期 `code/path`：

1. 缺少 tasks；
2. 未知顶层字段；
3. workload 类型错误；
4. 非有限 speed；
5. task ID 不连续；
6. resources 为空；
7. bandwidth 矩阵维度错误；
8. 边端点越界；
9. 重复边；
10. 环图；
11. 负 latency；
12. 未知 resource kind。

测试还覆盖嵌套未知字段、`NaN/Infinity/-Infinity`、JSON 语法行列、UTF-8 BOM、错误编码、直接构造 NaN、保存/加载 hash 一致，以及 CLI 成功/失败退出码。12 个固定案例中，6 个由 JSON Schema 结构层直接拒绝，另外 6 个由运行时跨字段不变量拒绝；两层合计全部拒绝。

复现：

```powershell
python -m pytest -o addopts="" -q tests/test_scenario_schema.py
python -m pytest -o addopts="" -q
```

本地实测：新增 `25 passed`，完整套件 `123 passed`。

## 7. 产物与 SHA-256

| 产物 | SHA-256（提交前工作树） |
| --- | --- |
| `schemas/scenario.schema.json` | `cfa6a5250cc88154cf3e77b56bf2514fc1f1a5cac526524411419dbe454ce1cb` |
| `trisched/scenario.py` | `c461513434a87b4949eb03d62f476ce8f35c5a2a80839919947c5a927eb9f26f` |
| `tests/fixtures/invalid/scenario_cases.json` | `1c51ad84c32c02777a0b9ab02684fd137919c2fc0e21a716786e97dd69b1f953` |
| `tests/test_scenario_schema.py` | `10e7b6ee2f50cfb1e5bd2ae402078d398031875a5b0071d42611c9b921adbeef` |

实现提交：`36edfc7`（`feat(scenario): [P0-03] add schema and structured diagnostics`）。

## 8. 兼容性、限制与复核要求

- `ScenarioValidationError` 是 `ValueError` 子类，原有 `pytest.raises(ValueError)` 和调用方仍兼容。
- 输入现在严格拒绝未知字段、空名称和非 device/edge/cloud 的 resource kind；这是有意的契约收紧。
- JSON Schema 无法直接表达“矩阵边长等于资源数”“ID 连续”“边端点范围”和 DAG 无环，不能只运行 schema 后跳过 `Scenario.from_dict`。
- 当前没有 schema 版本字段；不兼容扩展前必须先设计 `format_version` 迁移规则，而不是放开 `additionalProperties`。

成员 A 已在不读取 B 的非法 fixture、不调用场景生成器的前提下，独立构造缺字段、嵌套未知字段、bool 冒充整数、`1e309`、矩阵行列错误、重复边、环图和错误编码；Python API 与真实 CLI 子进程的 code/path、JSON 错误对象及退出码 0/2 均符合契约。完整测试为 `139 passed`，实现基线的六环境 CI 全部成功。复核方法和证据见 [P0-03 独立复核记录](./P0-03独立复核记录.md)。
