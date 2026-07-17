from __future__ import annotations

import csv
import json
import math
import platform
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .env import ScheduleResult, validate_schedule
from .learning import MaskedMLPPolicy
from .oracle import validate_schedule_independent
from .scenario import Scenario
from .schedulers import SchedulerRunner, build_scheduler_runners


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _policy_metrics(
    rows: list[dict[str, Any]], policy_name: str
) -> dict[str, float | int]:
    makespans = [float(row[f"{policy_name}_makespan"]) for row in rows]
    ratios = [float(row[f"{policy_name}_ratio"]) for row in rows]
    runtimes = [float(row[f"{policy_name}_runtime_ms"]) for row in rows]
    std = statistics.pstdev(ratios) if len(ratios) > 1 else 0.0
    ci_half_width = 1.96 * std / math.sqrt(len(ratios))
    return {
        "count": len(ratios),
        "mean_makespan": statistics.mean(makespans),
        "mean_ratio": statistics.mean(ratios),
        "ratio_std": std,
        "ratio_ci95_low": statistics.mean(ratios) - ci_half_width,
        "ratio_ci95_high": statistics.mean(ratios) + ci_half_width,
        "median_ratio": statistics.median(ratios),
        "p95_ratio": _percentile(ratios, 0.95),
        "win_rate_vs_heft": sum(ratio < 1.0 - 1e-9 for ratio in ratios)
        / len(ratios),
        "tie_rate_vs_heft": sum(abs(ratio - 1.0) <= 1e-9 for ratio in ratios)
        / len(ratios),
        "mean_runtime_ms": statistics.mean(runtimes),
        "valid_schedule_rate": 1.0,
    }


def evaluate_schedulers(
    scenarios: list[Scenario],
    schedulers: Sequence[SchedulerRunner],
    split_name: str,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate in-process and external schedulers through one validated path."""

    if not scenarios:
        raise ValueError("cannot evaluate an empty scenario split")
    names = [scheduler.name for scheduler in schedulers]
    if not names:
        raise ValueError("cannot evaluate an empty scheduler set")
    if len(names) != len(set(names)):
        raise ValueError("scheduler names must be unique")
    if "heft" not in names:
        raise ValueError("the unified evaluator requires HEFT as its baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    example_results: dict[str, ScheduleResult] = {}
    for index, scenario in enumerate(scenarios):
        results: dict[str, ScheduleResult] = {}
        runtimes: dict[str, float] = {}
        for scheduler in schedulers:
            start = time.perf_counter()
            result = scheduler.schedule(scenario)
            runtimes[scheduler.name] = (time.perf_counter() - start) * 1000.0
            if result.policy_name != scheduler.name:
                raise ValueError(
                    f"scheduler {scheduler.name} returned policy_name "
                    f"{result.policy_name!r}"
                )
            validate_schedule(scenario, result)
            validate_schedule_independent(scenario, result)
            results[scheduler.name] = result
        heft_makespan = results["heft"].makespan
        row: dict[str, Any] = {
            "split": split_name,
            "scenario_id": scenario.id,
            "scenario_hash": scenario.content_hash(),
            "task_count": scenario.task_count,
            "resource_count": scenario.resource_count,
            "edge_count": len(scenario.edges),
        }
        for name, result in results.items():
            row[f"{name}_makespan"] = result.makespan
            row[f"{name}_ratio"] = result.makespan / heft_makespan
            row[f"{name}_runtime_ms"] = runtimes[name]
        rows.append(row)
        if index == 0:
            example_results = results

    fieldnames = list(rows[0].keys())
    csv_path = output_dir / f"{split_name}_per_instance.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if example_results:
        example_path = output_dir / f"{split_name}_example_schedule.json"
        example_path.write_text(
            json.dumps(
                {name: result.to_dict() for name, result in example_results.items()},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    metrics = {name: _policy_metrics(rows, name) for name in names}
    return metrics, rows


def evaluate_split(
    scenarios: list[Scenario],
    learned_policy: MaskedMLPPolicy,
    split_name: str,
    output_dir: Path,
    random_seed: int,
    scheduler_specs: Sequence[str | Mapping[str, Any]] | None = None,
    config_dir: str | Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    schedulers = build_scheduler_runners(
        learned_policy=learned_policy,
        random_seed=random_seed,
        scheduler_specs=scheduler_specs,
        config_dir=config_dir,
    )
    return evaluate_schedulers(scenarios, schedulers, split_name, output_dir)


def dataset_manifest(scenarios: list[Scenario], split_name: str) -> dict[str, Any]:
    return {
        "split": split_name,
        "count": len(scenarios),
        "scenario_ids": [scenario.id for scenario in scenarios],
        "scenario_hashes": [scenario.content_hash() for scenario in scenarios],
        "task_counts": [scenario.task_count for scenario in scenarios],
    }


def write_summary(
    output_dir: Path,
    config: dict[str, Any],
    training_history: dict[str, Any],
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    manifests: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "format_version": 1,
        "primary_metric": {
            "name": "mean_ratio",
            "definition": "mean(policy_makespan / HEFT_makespan)",
            "lower_is_better": True,
            "split": "test",
            "policy": "masked_mlp",
            "value": test_metrics["masked_mlp"]["mean_ratio"],
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "training": training_history,
        "datasets": manifests,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "config": config,
    }
    path = output_dir / "summary.json"
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path
