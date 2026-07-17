from __future__ import annotations

import csv
import json
import math
import platform
import statistics
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .env import ScheduleResult, validate_schedule
from .learning import MaskedMLPPolicy
from .oracle import validate_schedule_independent
from .scenario import Scenario
from .schedulers import (
    SchedulerAdapterError,
    SchedulerRunner,
    build_scheduler_runners,
)


DEFAULT_FAILURE_PENALTY_RATIO = 10.0


def resolve_failure_penalty_ratio(value: Any) -> float:
    """Validate and normalize the ratio assigned to a failed scheduler attempt."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("evaluation.failure_penalty_ratio must be a finite number > 1")
    ratio = float(value)
    if not math.isfinite(ratio) or ratio <= 1.0:
        raise ValueError("evaluation.failure_penalty_ratio must be a finite number > 1")
    return ratio


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
    rows: list[dict[str, Any]],
    policy_name: str,
    failure_penalty_ratio: float,
) -> dict[str, Any]:
    score_makespans = [
        float(row[f"{policy_name}_score_makespan"]) for row in rows
    ]
    score_ratios = [float(row[f"{policy_name}_score_ratio"]) for row in rows]
    runtimes = [float(row[f"{policy_name}_runtime_ms"]) for row in rows]
    successful = [
        row for row in rows if row[f"{policy_name}_status"] == "success"
    ]
    failures = [row for row in rows if row[f"{policy_name}_status"] == "failure"]
    success_makespans = [
        float(row[f"{policy_name}_makespan"]) for row in successful
    ]
    success_ratios = [float(row[f"{policy_name}_ratio"]) for row in successful]
    std = statistics.pstdev(score_ratios) if len(score_ratios) > 1 else 0.0
    ci_half_width = 1.96 * std / math.sqrt(len(score_ratios))
    error_counts = Counter(
        str(row[f"{policy_name}_error_code"]) for row in failures
    )
    count = len(rows)
    success_count = len(successful)
    failure_count = len(failures)
    return {
        "count": count,
        "success_count": success_count,
        "failure_count": failure_count,
        "failure_rate": failure_count / count,
        "valid_schedule_rate": success_count / count,
        "failure_penalty_ratio": failure_penalty_ratio,
        "mean_makespan": statistics.mean(score_makespans),
        "mean_ratio": statistics.mean(score_ratios),
        "success_mean_makespan": (
            statistics.mean(success_makespans) if success_makespans else None
        ),
        "success_mean_ratio": (
            statistics.mean(success_ratios) if success_ratios else None
        ),
        "ratio_std": std,
        "ratio_ci95_low": statistics.mean(score_ratios) - ci_half_width,
        "ratio_ci95_high": statistics.mean(score_ratios) + ci_half_width,
        "median_ratio": statistics.median(score_ratios),
        "p95_ratio": _percentile(score_ratios, 0.95),
        "win_rate_vs_heft": sum(
            ratio < 1.0 - 1e-9 for ratio in score_ratios
        )
        / count,
        "tie_rate_vs_heft": sum(
            abs(ratio - 1.0) <= 1e-9 for ratio in score_ratios
        )
        / count,
        "loss_rate_vs_heft": sum(
            ratio > 1.0 + 1e-9 for ratio in score_ratios
        )
        / count,
        "mean_runtime_ms": statistics.mean(runtimes),
        "error_counts": dict(sorted(error_counts.items())),
    }


def _run_scheduler(
    scenario: Scenario, scheduler: SchedulerRunner
) -> tuple[ScheduleResult | None, float, Exception | None]:
    start = time.perf_counter()
    try:
        result = scheduler.schedule(scenario)
        if result.policy_name != scheduler.name:
            raise SchedulerAdapterError(
                "scheduler_invalid_response",
                "scheduler returned a mismatched policy_name",
                scheduler=scheduler.name,
                scenario_id=scenario.id,
                details={"actual_name": result.policy_name},
            )
        try:
            validate_schedule(scenario, result)
            validate_schedule_independent(scenario, result)
        except ValueError as error:
            raise SchedulerAdapterError(
                "scheduler_invalid_schedule",
                "scheduler returned an invalid schedule",
                scheduler=scheduler.name,
                scenario_id=scenario.id,
                details={"reason": str(error)},
            ) from error
    except Exception as error:
        return None, (time.perf_counter() - start) * 1000.0, error
    return result, (time.perf_counter() - start) * 1000.0, None


def _failure_payload(
    error: Exception, scheduler_name: str, scenario_id: str
) -> dict[str, Any]:
    if isinstance(error, SchedulerAdapterError):
        payload = error.to_dict()
        payload.setdefault("scheduler", scheduler_name)
        payload.setdefault("scenario_id", scenario_id)
        return payload
    message = str(error).strip() or type(error).__name__
    return {
        "code": "scheduler_execution_failed",
        "message": message,
        "scheduler": scheduler_name,
        "scenario_id": scenario_id,
        "details": {"exception_type": type(error).__name__},
    }


def evaluate_schedulers(
    scenarios: list[Scenario],
    schedulers: Sequence[SchedulerRunner],
    split_name: str,
    output_dir: Path,
    failure_penalty_ratio: float = DEFAULT_FAILURE_PENALTY_RATIO,
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
    failure_penalty_ratio = resolve_failure_penalty_ratio(failure_penalty_ratio)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    example_results: dict[str, ScheduleResult] = {}
    failure_records: list[dict[str, Any]] = []
    heft_runner = next(scheduler for scheduler in schedulers if scheduler.name == "heft")
    for scenario in scenarios:
        outcomes: dict[str, dict[str, Any]] = {}
        heft_result, heft_runtime, heft_error = _run_scheduler(
            scenario, heft_runner
        )
        if heft_error is not None:
            cause = _failure_payload(heft_error, "heft", scenario.id)
            raise SchedulerAdapterError(
                "scheduler_baseline_failed",
                "HEFT baseline failed; per-instance ratios are undefined",
                scheduler="heft",
                scenario_id=scenario.id,
                details={"cause": cause},
            ) from heft_error
        assert heft_result is not None
        outcomes["heft"] = {
            "result": heft_result,
            "runtime_ms": heft_runtime,
            "error": None,
        }
        example_results.setdefault("heft", heft_result)

        for scheduler in schedulers:
            if scheduler.name == "heft":
                continue
            result, runtime_ms, error = _run_scheduler(scenario, scheduler)
            if error is not None:
                outcomes[scheduler.name] = {
                    "result": None,
                    "runtime_ms": runtime_ms,
                    "error": _failure_payload(
                        error, scheduler.name, scenario.id
                    ),
                }
            else:
                assert result is not None
                outcomes[scheduler.name] = {
                    "result": result,
                    "runtime_ms": runtime_ms,
                    "error": None,
                }
                example_results.setdefault(scheduler.name, result)

        heft_makespan = heft_result.makespan
        row: dict[str, Any] = {
            "split": split_name,
            "scenario_id": scenario.id,
            "scenario_hash": scenario.content_hash(),
            "task_count": scenario.task_count,
            "resource_count": scenario.resource_count,
            "edge_count": len(scenario.edges),
        }
        for name in names:
            outcome = outcomes[name]
            result = outcome["result"]
            error = outcome["error"]
            runtime_ms = float(outcome["runtime_ms"])
            if result is not None:
                ratio = result.makespan / heft_makespan
                row[f"{name}_status"] = "success"
                row[f"{name}_makespan"] = result.makespan
                row[f"{name}_ratio"] = ratio
                row[f"{name}_score_makespan"] = result.makespan
                row[f"{name}_score_ratio"] = ratio
                row[f"{name}_penalty_applied"] = False
                row[f"{name}_error_code"] = ""
                row[f"{name}_error_message"] = ""
                row[f"{name}_error_details"] = ""
            else:
                penalty_makespan = failure_penalty_ratio * heft_makespan
                row[f"{name}_status"] = "failure"
                row[f"{name}_makespan"] = ""
                row[f"{name}_ratio"] = ""
                row[f"{name}_score_makespan"] = penalty_makespan
                row[f"{name}_score_ratio"] = failure_penalty_ratio
                row[f"{name}_penalty_applied"] = True
                row[f"{name}_error_code"] = error["code"]
                row[f"{name}_error_message"] = error["message"]
                row[f"{name}_error_details"] = json.dumps(
                    error.get("details", {}),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                failure_records.append(
                    {
                        "format_version": 1,
                        "split": split_name,
                        "scenario_id": scenario.id,
                        "scenario_hash": scenario.content_hash(),
                        "scheduler": name,
                        "runtime_ms": runtime_ms,
                        "failure_penalty_ratio": failure_penalty_ratio,
                        "score_makespan": penalty_makespan,
                        "error": error,
                    }
                )
            row[f"{name}_runtime_ms"] = runtime_ms
        rows.append(row)

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
    failures_path = output_dir / f"{split_name}_failures.jsonl"
    with failures_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in failure_records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    metrics = {
        name: _policy_metrics(rows, name, failure_penalty_ratio)
        for name in names
    }
    return metrics, rows


def evaluate_split(
    scenarios: list[Scenario],
    learned_policy: MaskedMLPPolicy,
    split_name: str,
    output_dir: Path,
    random_seed: int,
    scheduler_specs: Sequence[str | Mapping[str, Any]] | None = None,
    config_dir: str | Path | None = None,
    failure_penalty_ratio: float = DEFAULT_FAILURE_PENALTY_RATIO,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    schedulers = build_scheduler_runners(
        learned_policy=learned_policy,
        random_seed=random_seed,
        scheduler_specs=scheduler_specs,
        config_dir=config_dir,
    )
    return evaluate_schedulers(
        scenarios,
        schedulers,
        split_name,
        output_dir,
        failure_penalty_ratio,
    )


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
        "format_version": 2,
        "primary_metric": {
            "name": "mean_ratio",
            "definition": (
                "mean(score_ratio); successful score_ratio="
                "policy_makespan/HEFT_makespan, failed score_ratio="
                "failure_penalty_ratio"
            ),
            "lower_is_better": True,
            "split": "test",
            "policy": "masked_mlp",
            "value": test_metrics["masked_mlp"]["mean_ratio"],
        },
        "scoring": {
            "failure_penalty_ratio": config["evaluation"][
                "failure_penalty_ratio"
            ],
            "failures_remain_in_denominator": True,
            "publishable_requires_zero_failures": True,
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
        "run_manifest": "run_manifest.json",
    }
    path = output_dir / "summary.json"
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path
