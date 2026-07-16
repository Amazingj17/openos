from __future__ import annotations

import csv
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .env import ScheduleResult, run_policy
from .learning import MaskedMLPPolicy
from .policies import GreedyEarliestFinishPolicy, HeftPolicy, RandomPolicy
from .scenario import Scenario


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


def evaluate_split(
    scenarios: list[Scenario],
    learned_policy: MaskedMLPPolicy,
    split_name: str,
    output_dir: Path,
    random_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    example_results: dict[str, ScheduleResult] = {}
    for index, scenario in enumerate(scenarios):
        policies = [
            HeftPolicy(),
            GreedyEarliestFinishPolicy(),
            RandomPolicy(seed=random_seed),
            learned_policy,
        ]
        results: dict[str, ScheduleResult] = {}
        runtimes: dict[str, float] = {}
        for policy in policies:
            start = time.perf_counter()
            result = run_policy(scenario, policy)
            runtimes[policy.name] = (time.perf_counter() - start) * 1000.0
            results[policy.name] = result
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
    metrics = {
        name: _policy_metrics(rows, name)
        for name in ("heft", "greedy_eft", "random", "masked_mlp")
    }
    return metrics, rows


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
