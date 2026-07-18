from __future__ import annotations

import html
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


_MLP_COLOR = "#2563eb"
_GNN_COLOR = "#f97316"
_WIN_COLOR = "#16a34a"
_LOSS_COLOR = "#dc2626"
_TIE_COLOR = "#737373"


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be an array")
    return value


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _svg_text(
    x: float,
    y: float,
    value: str,
    *,
    anchor: str = "start",
    css_class: str = "label",
) -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
        f'class="{css_class}">{html.escape(value)}</text>'
    )


def _comparison_data(report: Mapping[str, Any]) -> dict[str, Any]:
    paired = _mapping(report.get("paired_validation"), "paired_validation")
    bootstrap = _mapping(
        paired.get("hierarchical_paired_bootstrap"),
        "paired_validation.hierarchical_paired_bootstrap",
    )
    per_seed_raw = _sequence(report.get("per_seed"), "per_seed")
    per_scenario_raw = _sequence(
        report.get("per_scenario_seed_mean"),
        "per_scenario_seed_mean",
    )
    if not per_seed_raw or not per_scenario_raw:
        raise ValueError("comparison requires per-seed and per-scenario rows")

    per_seed = []
    for index, raw in enumerate(per_seed_raw):
        item = _mapping(raw, f"per_seed[{index}]")
        pairs = _mapping(item.get("pairs"), f"per_seed[{index}].pairs")
        per_seed.append(
            {
                "seed": _integer(item.get("seed"), f"per_seed[{index}].seed"),
                "mlp": _number(
                    item.get("masked_mlp_mean_ratio"),
                    f"per_seed[{index}].masked_mlp_mean_ratio",
                ),
                "gnn": _number(
                    item.get("task_gnn_mean_ratio"),
                    f"per_seed[{index}].task_gnn_mean_ratio",
                ),
                "mlp_p95": _number(
                    item.get("masked_mlp_p95_ratio"),
                    f"per_seed[{index}].masked_mlp_p95_ratio",
                ),
                "gnn_p95": _number(
                    item.get("task_gnn_p95_ratio"),
                    f"per_seed[{index}].task_gnn_p95_ratio",
                ),
                "wins": _integer(
                    pairs.get("task_gnn_win"),
                    f"per_seed[{index}].pairs.task_gnn_win",
                ),
                "ties": _integer(
                    pairs.get("tie"),
                    f"per_seed[{index}].pairs.tie",
                ),
                "losses": _integer(
                    pairs.get("mlp_win"),
                    f"per_seed[{index}].pairs.mlp_win",
                ),
            }
        )

    per_scenario = []
    for index, raw in enumerate(per_scenario_raw):
        item = _mapping(raw, f"per_scenario_seed_mean[{index}]")
        scenario_id = item.get("scenario_id")
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError(f"per_scenario_seed_mean[{index}].scenario_id is invalid")
        per_scenario.append(
            {
                "scenario_id": scenario_id,
                "delta": _number(
                    item.get("task_gnn_minus_mlp"),
                    f"per_scenario_seed_mean[{index}].task_gnn_minus_mlp",
                ),
            }
        )

    counts = _mapping(paired.get("scenario_mean_pairs"), "scenario_mean_pairs")
    return {
        "mlp_mean": _number(
            paired.get("masked_mlp_mean_ratio"),
            "paired_validation.masked_mlp_mean_ratio",
        ),
        "gnn_mean": _number(
            paired.get("task_gnn_mean_ratio"),
            "paired_validation.task_gnn_mean_ratio",
        ),
        "delta": _number(
            paired.get("mean_paired_delta"),
            "paired_validation.mean_paired_delta",
        ),
        "relative_change": _number(
            paired.get("relative_mean_change"),
            "paired_validation.relative_mean_change",
        ),
        "ci_low": _number(bootstrap.get("lower"), "bootstrap.lower"),
        "ci_high": _number(bootstrap.get("upper"), "bootstrap.upper"),
        "ci_excludes_zero": bool(bootstrap.get("excludes_zero_in_task_gnn_direction")),
        "scenario_wins": _integer(counts.get("task_gnn_win"), "scenario wins"),
        "scenario_ties": _integer(counts.get("tie"), "scenario ties"),
        "scenario_losses": _integer(counts.get("mlp_win"), "scenario losses"),
        "per_seed": per_seed,
        "per_scenario": sorted(per_scenario, key=lambda row: row["delta"]),
        "gate_passed": bool(report.get("development_gate_passed")),
        "parameters": _mapping(report.get("parameters"), "parameters"),
        "training_wall_seconds": _mapping(
            report.get("training_wall_seconds", {}),
            "training_wall_seconds",
        ),
        "cpu_latency": report.get("cpu_latency"),
    }


def _chart_svg(data: Mapping[str, Any]) -> str:
    width = 1080
    height = 720
    plot_left = 82.0
    plot_right = 1038.0
    seed_top = 86.0
    seed_bottom = 350.0
    seed_rows = data["per_seed"]
    y_max = max(
        1.05,
        max(max(float(item["mlp"]), float(item["gnn"])) for item in seed_rows) * 1.15,
    )
    y_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    if y_max > 1.05:
        y_ticks.append(math.ceil(y_max * 4.0) / 4.0)
    y_ticks = sorted({tick for tick in y_ticks if tick <= y_max + 1e-9})

    def seed_y(value: float) -> float:
        return seed_bottom - value / y_max * (seed_bottom - seed_top)

    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        'role="img" aria-labelledby="chart-title chart-desc">',
        '<title id="chart-title">Masked MLP 与 Task-GNN validation 性能比较</title>',
        '<desc id="chart-desc">上图比较各随机种子的平均 makespan ratio；下图展示每个场景的 Task-GNN 减 Masked MLP ratio，负数代表 Task-GNN 更好。</desc>',
        "<style>",
        ":root{color-scheme:light dark}",
        ".bg{fill:#ffffff}.fg{fill:#171717}.muted{fill:#525252}.grid{stroke:#d4d4d4;stroke-width:1}.axis{stroke:#737373;stroke-width:1.2}.baseline{stroke:#737373;stroke-width:1.5;stroke-dasharray:6 5}.zero{stroke:#171717;stroke-width:1.5}.label{font:14px system-ui,sans-serif;fill:#171717}.small{font:12px system-ui,sans-serif;fill:#525252}.heading{font:500 18px system-ui,sans-serif;fill:#171717}.mlp{fill:#2563eb}.gnn{fill:#f97316}.win{fill:#16a34a}.loss{fill:#dc2626}.tie{fill:#737373}",
        "@media(prefers-color-scheme:dark){.bg{fill:#171717}.fg{fill:#fafafa}.muted{fill:#d4d4d4}.grid{stroke:#404040}.axis{stroke:#a3a3a3}.baseline{stroke:#a3a3a3}.zero{stroke:#fafafa}.label,.heading{fill:#fafafa}.small{fill:#d4d4d4}.mlp{fill:#60a5fa}.gnn{fill:#fb923c}.win{fill:#4ade80}.loss{fill:#f87171}.tie{fill:#a3a3a3}}",
        "</style>",
        f'<rect class="bg" width="{width}" height="{height}"/>',
        _svg_text(
            plot_left, 38, "各 seed 的 validation 平均 ratio（越低越好）", css_class="heading"
        ),
    ]

    for tick in y_ticks:
        y = seed_y(tick)
        chunks.append(
            f'<line class="grid" x1="{plot_left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}"/>'
        )
        chunks.append(
            _svg_text(
                plot_left - 10, y + 4, _fmt(tick, 2), anchor="end", css_class="small"
            )
        )
    baseline_y = seed_y(1.0)
    chunks.append(
        f'<line class="baseline" x1="{plot_left}" y1="{baseline_y:.2f}" x2="{plot_right}" y2="{baseline_y:.2f}"/>'
    )
    chunks.append(
        _svg_text(
            plot_right, baseline_y - 7, "HEFT = 1.0", anchor="end", css_class="small"
        )
    )
    chunks.append(
        f'<line class="axis" x1="{plot_left}" y1="{seed_bottom}" x2="{plot_right}" y2="{seed_bottom}"/>'
    )

    group_width = (plot_right - plot_left) / len(seed_rows)
    bar_width = min(56.0, group_width * 0.24)
    for index, item in enumerate(seed_rows):
        center = plot_left + group_width * (index + 0.5)
        mlp_x = center - bar_width - 4
        gnn_x = center + 4
        mlp_y = seed_y(float(item["mlp"]))
        gnn_y = seed_y(float(item["gnn"]))
        chunks.extend(
            [
                f'<rect class="mlp" x="{mlp_x:.2f}" y="{mlp_y:.2f}" width="{bar_width:.2f}" height="{seed_bottom - mlp_y:.2f}"/>',
                f'<rect class="gnn" x="{gnn_x:.2f}" y="{gnn_y:.2f}" width="{bar_width:.2f}" height="{seed_bottom - gnn_y:.2f}"/>',
                _svg_text(
                    mlp_x + bar_width / 2,
                    mlp_y - 7,
                    _fmt(float(item["mlp"]), 3),
                    anchor="middle",
                    css_class="small",
                ),
                _svg_text(
                    gnn_x + bar_width / 2,
                    gnn_y - 7,
                    _fmt(float(item["gnn"]), 3),
                    anchor="middle",
                    css_class="small",
                ),
                _svg_text(
                    center,
                    seed_bottom + 24,
                    str(item["seed"]),
                    anchor="middle",
                    css_class="small",
                ),
            ]
        )

    legend_y = 66
    chunks.extend(
        [
            f'<rect class="mlp" x="{plot_right - 245}" y="{legend_y - 12}" width="14" height="14"/>',
            _svg_text(plot_right - 224, legend_y, "Masked MLP", css_class="small"),
            f'<rect class="gnn" x="{plot_right - 115}" y="{legend_y - 12}" width="14" height="14"/>',
            _svg_text(plot_right - 94, legend_y, "Task-GNN", css_class="small"),
        ]
    )

    delta_top = 455.0
    delta_bottom = 645.0
    zero_y = (delta_top + delta_bottom) / 2
    scenario_rows = data["per_scenario"]
    max_abs_delta = max(max(abs(float(item["delta"])) for item in scenario_rows), 0.01)
    delta_scale = (delta_bottom - delta_top) * 0.43 / max_abs_delta
    chunks.extend(
        [
            _svg_text(
                plot_left,
                414,
                "逐场景 seed 均值差：Task-GNN − Masked MLP",
                css_class="heading",
            ),
            _svg_text(
                plot_left,
                437,
                "负值（绿色）表示 Task-GNN 更好；正值（红色）表示 Masked MLP 更好",
                css_class="small",
            ),
            f'<line class="zero" x1="{plot_left}" y1="{zero_y:.2f}" x2="{plot_right}" y2="{zero_y:.2f}"/>',
        ]
    )
    scenario_width = (plot_right - plot_left) / len(scenario_rows)
    mark_width = max(2.0, scenario_width * 0.68)
    for index, item in enumerate(scenario_rows):
        delta = float(item["delta"])
        x = plot_left + scenario_width * index + (scenario_width - mark_width) / 2
        bar_height = abs(delta) * delta_scale
        y = zero_y - bar_height if delta >= 0 else zero_y
        css_class = "loss" if delta > 1e-9 else "win" if delta < -1e-9 else "tie"
        chunks.append(
            f'<rect class="{css_class}" x="{x:.2f}" y="{y:.2f}" width="{mark_width:.2f}" height="{max(bar_height, 1.5):.2f}"><title>{html.escape(item["scenario_id"])}: {delta:+.4f}</title></rect>'
        )
    chunks.extend(
        [
            _svg_text(
                plot_left - 10,
                delta_top + 5,
                f"+{max_abs_delta:.3f}",
                anchor="end",
                css_class="small",
            ),
            _svg_text(plot_left - 10, zero_y + 4, "0", anchor="end", css_class="small"),
            _svg_text(
                plot_left - 10,
                delta_bottom,
                f"−{max_abs_delta:.3f}",
                anchor="end",
                css_class="small",
            ),
            _svg_text(
                plot_right,
                690,
                (
                    f"场景胜/平/负：{data['scenario_wins']} / "
                    f"{data['scenario_ties']} / {data['scenario_losses']}"
                ),
                anchor="end",
                css_class="label",
            ),
            "</svg>",
        ]
    )
    return "\n".join(chunks) + "\n"


def _optional_number(value: Any) -> float | None:
    if value is None:
        return None
    return _number(value, "optional metric")


def _efficiency_rows(data: Mapping[str, Any]) -> list[tuple[str, str, str, str]]:
    parameters = data["parameters"]
    mlp_parameters = _sequence(parameters.get("masked_mlp"), "parameters.masked_mlp")
    gnn_parameters = _sequence(parameters.get("task_gnn"), "parameters.task_gnn")
    mlp_first = _mapping(mlp_parameters[0], "parameters.masked_mlp[0]")
    gnn_first = _mapping(gnn_parameters[0], "parameters.task_gnn[0]")
    rows = [
        (
            "参数量",
            str(_integer(mlp_first.get("parameter_count"), "MLP parameter_count")),
            str(_integer(gnn_first.get("parameter_count"), "GNN parameter_count")),
            "越少越轻量",
        )
    ]
    training = data["training_wall_seconds"]
    mlp_wall = _optional_number(training.get("masked_mlp"))
    gnn_wall = _optional_number(training.get("task_gnn"))
    if mlp_wall is not None and gnn_wall is not None:
        rows.append(("端到端训练", f"{mlp_wall:.1f} s", f"{gnn_wall:.1f} s", "越低越好"))
    latency = data["cpu_latency"]
    if latency is not None:
        latency_map = _mapping(latency, "cpu_latency")
        mlp_latency = _mapping(latency_map.get("masked_mlp"), "cpu_latency.masked_mlp")
        gnn_latency = _mapping(latency_map.get("task_gnn"), "cpu_latency.task_gnn")
        rows.append(
            (
                "CPU 推理 P50",
                f"{_number(mlp_latency.get('p50_ms'), 'MLP p50_ms'):.2f} ms",
                f"{_number(gnn_latency.get('p50_ms'), 'GNN p50_ms'):.2f} ms",
                "含 run_policy 与双校验器",
            )
        )
        rows.append(
            (
                "CPU 推理 P95",
                f"{_number(mlp_latency.get('p95_ms'), 'MLP p95_ms'):.2f} ms",
                f"{_number(gnn_latency.get('p95_ms'), 'GNN p95_ms'):.2f} ms",
                "越低越好",
            )
        )
    return rows


def _html_report(
    report: Mapping[str, Any],
    data: Mapping[str, Any],
    svg_name: str,
    json_name: str,
    per_instance_name: str,
    per_seed_name: str,
    per_scenario_name: str,
) -> str:
    ci_status = "区间排除 0" if data["ci_excludes_zero"] else "区间跨 0"
    gate_status = "通过" if data["gate_passed"] else "未通过"
    conclusion = (
        "Task-GNN 的配对改善通过当前 validation 门禁。"
        if data["gate_passed"]
        else "Task-GNN 的点估计可改善，但当前配对证据不足以替换 Masked MLP。"
    )
    seed_rows = []
    for item in data["per_seed"]:
        seed_rows.append(
            "<tr>"
            f"<td>{item['seed']}</td>"
            f"<td>{_fmt(float(item['mlp']))}</td>"
            f"<td>{_fmt(float(item['gnn']))}</td>"
            f"<td>{float(item['gnn']) - float(item['mlp']):+.4f}</td>"
            f"<td>{_fmt(float(item['mlp_p95']))}</td>"
            f"<td>{_fmt(float(item['gnn_p95']))}</td>"
            f"<td>{item['wins']} / {item['ties']} / {item['losses']}</td>"
            "</tr>"
        )
    efficiency_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(metric)}</td>"
        f"<td>{html.escape(mlp)}</td>"
        f"<td>{html.escape(gnn)}</td>"
        f"<td>{html.escape(note)}</td>"
        "</tr>"
        for metric, mlp, gnn, note in _efficiency_rows(data)
    )
    escaped_files = [
        (json_name, "完整 JSON"),
        (per_instance_name, "逐 seed × 场景 CSV"),
        (per_seed_name, "逐 seed CSV"),
        (per_scenario_name, "逐场景 CSV"),
        (svg_name, "独立 SVG 图"),
    ]
    file_links = "".join(
        f'<a href="{html.escape(name)}">{html.escape(label)}</a>'
        for name, label in escaped_files
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Masked MLP 与 Task-GNN 性能比较</title>
  <style>
    :root {{ color-scheme: light dark; --bg:#f7f7f8; --fg:#171717; --muted:#5f5f66; --card:#ffffff; --border:#d9d9df; --mlp:{_MLP_COLOR}; --gnn:{_GNN_COLOR}; }}
    @media (prefers-color-scheme: dark) {{ :root {{ --bg:#111113; --fg:#f5f5f5; --muted:#b8b8c0; --card:#1b1b1f; --border:#3b3b43; --mlp:#60a5fa; --gnn:#fb923c; }} }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--fg); font-family:system-ui,-apple-system,"Segoe UI",sans-serif; line-height:1.55; }}
    main {{ max-width:1180px; margin:0 auto; padding:28px 20px 48px; }}
    h1,h2 {{ font-weight:500; line-height:1.25; }}
    h1 {{ margin:0 0 8px; }} h2 {{ margin:34px 0 14px; }}
    .muted {{ color:var(--muted); }}
    .summary {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; margin:24px 0; }}
    .card {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:16px; }}
    .value {{ font-size:1.65rem; font-variant-numeric:tabular-nums; margin-top:5px; }}
    .mlp {{ color:var(--mlp); }} .gnn {{ color:var(--gnn); }}
    figure {{ margin:18px 0 0; }} figure img {{ display:block; width:100%; height:auto; background:var(--card); border:1px solid var(--border); border-radius:12px; }}
    figcaption {{ color:var(--muted); margin-top:8px; }}
    .table-wrap {{ overflow-x:auto; border:1px solid var(--border); border-radius:12px; background:var(--card); }}
    table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }}
    th,td {{ padding:10px 12px; border-bottom:1px solid var(--border); text-align:right; white-space:nowrap; }}
    th:first-child,td:first-child {{ text-align:left; }} tr:last-child td {{ border-bottom:0; }}
    th {{ color:var(--muted); font-weight:500; }}
    .files {{ display:flex; flex-wrap:wrap; gap:10px; }}
    .files a {{ color:var(--fg); border:1px solid var(--border); border-radius:8px; padding:8px 11px; text-decoration:none; background:var(--card); }}
    .files a:hover {{ border-color:var(--fg); }}
    @media (max-width:720px) {{ .summary {{ grid-template-columns:1fr; }} main {{ padding:20px 12px 36px; }} }}
  </style>
</head>
<body>
<main>
  <h1>Masked MLP 与 Task-GNN 性能比较</h1>
  <div class="muted">validation-only；ratio = model makespan / HEFT makespan，越低越好。</div>
  <section class="summary" aria-label="性能摘要">
    <div class="card"><div class="muted">Masked MLP 平均 ratio</div><div class="value mlp">{_fmt(float(data['mlp_mean']))}</div></div>
    <div class="card"><div class="muted">Task-GNN 平均 ratio</div><div class="value gnn">{_fmt(float(data['gnn_mean']))}</div></div>
    <div class="card"><div class="muted">Task-GNN − MLP</div><div class="value">{float(data['delta']):+.4f}</div><div class="muted">95% CI [{float(data['ci_low']):+.4f}, {float(data['ci_high']):+.4f}]，{ci_status}</div></div>
  </section>
  <p><strong>门禁：{gate_status}。</strong> {html.escape(conclusion)} 相对均值变化为 {float(data['relative_change']) * 100:+.2f}%。</p>
  <figure>
    <img src="{html.escape(svg_name)}" alt="各 seed 平均 ratio 柱状图和逐场景配对差值图">
    <figcaption>横向虚线为 HEFT=1.0；逐场景差值为 Task-GNN 减 Masked MLP。</figcaption>
  </figure>
  <h2>逐 seed 结果</h2>
  <div class="table-wrap"><table>
    <thead><tr><th>seed</th><th>MLP mean</th><th>GNN mean</th><th>差值</th><th>MLP P95</th><th>GNN P95</th><th>GNN 胜/平/负</th></tr></thead>
    <tbody>{''.join(seed_rows)}</tbody>
  </table></div>
  <h2>计算开销</h2>
  <div class="table-wrap"><table>
    <thead><tr><th>指标</th><th>Masked MLP</th><th>Task-GNN</th><th>口径</th></tr></thead>
    <tbody>{efficiency_rows}</tbody>
  </table></div>
  <h2>结果文件</h2>
  <div class="files">{file_links}</div>
  <p class="muted">公开 test 未访问；本报告只表示 validation 开发比较，不构成最终泛化结论。</p>
</main>
</body>
</html>
"""


def write_model_comparison_visualizations(
    report: Mapping[str, Any],
    *,
    html_path: str | Path,
    svg_path: str | Path,
    json_name: str,
    per_instance_name: str,
    per_seed_name: str,
    per_scenario_name: str,
) -> tuple[Path, Path]:
    """Write a standalone HTML report and SVG chart from comparison JSON data."""

    data = _comparison_data(report)
    html_destination = Path(html_path)
    svg_destination = Path(svg_path)
    html_destination.parent.mkdir(parents=True, exist_ok=True)
    svg_destination.parent.mkdir(parents=True, exist_ok=True)
    svg_destination.write_text(_chart_svg(data), encoding="utf-8", newline="\n")
    html_destination.write_text(
        _html_report(
            report,
            data,
            svg_destination.name,
            json_name,
            per_instance_name,
            per_seed_name,
            per_scenario_name,
        ),
        encoding="utf-8",
        newline="\n",
    )
    return html_destination, svg_destination


def load_and_write_model_comparison_visualizations(
    comparison_path: str | Path,
) -> tuple[Path, Path]:
    """Regenerate visual files next to an existing comparison JSON."""

    source = Path(comparison_path)
    report = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("comparison JSON must be an object")
    per_instance = source.with_name(source.stem + "_per_instance.csv")
    per_seed = source.with_name(source.stem + "_per_seed.csv")
    per_scenario = source.with_name(source.stem + "_per_scenario.csv")
    return write_model_comparison_visualizations(
        report,
        html_path=source.with_suffix(".html"),
        svg_path=source.with_suffix(".svg"),
        json_name=source.name,
        per_instance_name=per_instance.name,
        per_seed_name=per_seed.name,
        per_scenario_name=per_scenario.name,
    )
