from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from trisched.benchmark import load_frozen_split
from trisched.env import run_policy
from trisched.gnn import TaskGNNPolicy, task_gnn_parameter_hash
from trisched.learning import MaskedMLPPolicy
from trisched.bc import policy_parameter_hash


TOLERANCE = 1e-9


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def _outcome(delta: float) -> str:
    if delta < -TOLERANCE:
        return "task_gnn_win"
    if delta > TOLERANCE:
        return "mlp_win"
    return "tie"


def _counts(values: np.ndarray) -> dict[str, int]:
    outcomes = [_outcome(float(value)) for value in values.ravel()]
    return {
        "task_gnn_win": outcomes.count("task_gnn_win"),
        "tie": outcomes.count("tie"),
        "mlp_win": outcomes.count("mlp_win"),
    }


def _paired_statistics(
    mlp_ratios: np.ndarray,
    task_gnn_ratios: np.ndarray,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    mlp = np.asarray(mlp_ratios, dtype=np.float64)
    task_gnn = np.asarray(task_gnn_ratios, dtype=np.float64)
    if mlp.shape != task_gnn.shape or mlp.ndim != 2:
        raise ValueError("paired ratios must have the same seed-by-scenario shape")
    if mlp.shape[0] < 1 or mlp.shape[1] < 2:
        raise ValueError("paired ratios require seeds and at least two scenarios")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    delta = task_gnn - mlp
    scenario_delta = np.mean(delta, axis=0)
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(bootstrap_samples):
        seed_indices = rng.integers(0, mlp.shape[0], size=mlp.shape[0])
        scenario_indices = rng.integers(
            0,
            mlp.shape[1],
            size=mlp.shape[1],
        )
        bootstrap[index] = float(np.mean(delta[np.ix_(seed_indices, scenario_indices)]))
    lower, upper = np.percentile(bootstrap, [2.5, 97.5])
    return {
        "ratio_direction": "lower_is_better",
        "delta_definition": "task_gnn_ratio - masked_mlp_ratio",
        "seed_count": int(mlp.shape[0]),
        "scenario_count": int(mlp.shape[1]),
        "pair_count": int(delta.size),
        "masked_mlp_mean_ratio": float(np.mean(mlp)),
        "task_gnn_mean_ratio": float(np.mean(task_gnn)),
        "mean_paired_delta": float(np.mean(delta)),
        "median_paired_delta": float(np.median(delta)),
        "relative_mean_change": float(np.mean(task_gnn) / np.mean(mlp) - 1.0),
        "all_seed_scenario_pairs": _counts(delta),
        "scenario_mean_pairs": _counts(scenario_delta),
        "hierarchical_paired_bootstrap": {
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "resampling": (
                "sample seeds and shared scenario ids with replacement; "
                "average task-GNN minus MLP ratio"
            ),
            "confidence_level": 0.95,
            "lower": float(lower),
            "upper": float(upper),
            "excludes_zero_in_task_gnn_direction": bool(upper < 0.0),
        },
    }


def _summary_seed_map(summary: dict[str, Any]) -> dict[int, dict[str, Any]]:
    values = summary.get("seeds")
    if not isinstance(values, list):
        raise ValueError("summary seeds must be an array")
    result: dict[int, dict[str, Any]] = {}
    for item in values:
        if not isinstance(item, dict) or not isinstance(item.get("seed"), int):
            raise ValueError("summary seed entry is invalid")
        seed = int(item["seed"])
        if seed in result:
            raise ValueError(f"duplicate seed {seed}")
        result[seed] = item
    return result


def _diagnostic_rows(
    path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    diagnostic = _load_json(path)
    if diagnostic.get("split") != "validation":
        raise ValueError(f"diagnostic is not validation: {path}")
    if diagnostic.get("test_accessed") is not False:
        raise ValueError(f"diagnostic does not forbid test: {path}")
    best = diagnostic.get("best")
    if not isinstance(best, dict):
        raise ValueError(f"diagnostic best entry is invalid: {path}")
    metrics = best.get("metrics")
    rows = best.get("per_instance")
    if not isinstance(metrics, dict) or not isinstance(rows, list):
        raise ValueError(f"diagnostic best payload is invalid: {path}")
    mapped: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("scenario_id"), str):
            raise ValueError(f"diagnostic row is invalid: {path}")
        if row["scenario_id"] in mapped:
            raise ValueError(f"duplicate scenario in {path}: {row['scenario_id']}")
        mapped[row["scenario_id"]] = row
    return metrics, mapped


def _latency_summary(samples: list[float]) -> dict[str, Any]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "sample_count": int(values.size),
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
    }


def _benchmark_checkpoints(
    mlp_root: Path,
    task_gnn_root: Path,
    mlp_seeds: dict[int, dict[str, Any]],
    task_gnn_seeds: dict[int, dict[str, Any]],
    scenarios: list[Any],
    repeats: int,
) -> dict[str, Any] | None:
    if repeats <= 0:
        return None
    mlp_samples: list[float] = []
    task_gnn_samples: list[float] = []
    paired_ratios: list[float] = []
    for seed in sorted(mlp_seeds):
        mlp_name = mlp_seeds[seed]["best_checkpoint"]["actor"]["name"]
        task_gnn_name = task_gnn_seeds[seed]["best_checkpoint"]["actor"]["name"]
        mlp = MaskedMLPPolicy.load(mlp_root / mlp_name)
        task_gnn = TaskGNNPolicy.load(task_gnn_root / task_gnn_name)
        run_policy(scenarios[0], mlp)
        run_policy(scenarios[0], task_gnn)
        for repeat in range(repeats):
            for scenario_index, scenario in enumerate(scenarios):
                ordered = (
                    (("mlp", mlp), ("task_gnn", task_gnn))
                    if (repeat + scenario_index + seed) % 2 == 0
                    else (("task_gnn", task_gnn), ("mlp", mlp))
                )
                elapsed: dict[str, float] = {}
                for name, policy in ordered:
                    started = time.perf_counter_ns()
                    run_policy(scenario, policy)
                    elapsed[name] = (time.perf_counter_ns() - started) / 1_000_000.0
                mlp_samples.append(elapsed["mlp"])
                task_gnn_samples.append(elapsed["task_gnn"])
                paired_ratios.append(elapsed["task_gnn"] / max(elapsed["mlp"], 1e-12))
    mlp_summary = _latency_summary(mlp_samples)
    task_gnn_summary = _latency_summary(task_gnn_samples)
    return {
        "scope": (
            "best validation checkpoints; full run_policy plus validators; "
            "alternating paired order"
        ),
        "repeats_per_seed_scenario": repeats,
        "masked_mlp": mlp_summary,
        "task_gnn": task_gnn_summary,
        "task_gnn_over_mlp_p50": (task_gnn_summary["p50_ms"] / mlp_summary["p50_ms"]),
        "paired_latency_ratio_p50": float(np.percentile(paired_ratios, 50)),
        "paired_latency_ratio_p95": float(np.percentile(paired_ratios, 95)),
        "test_accessed": False,
    }


def compare(args: argparse.Namespace) -> Path:
    mlp_root = Path(args.mlp_root).resolve()
    task_gnn_root = Path(args.task_gnn_root).resolve()
    output = Path(args.output).resolve()
    csv_path = output.with_name(output.stem + "_per_instance.csv")
    mlp_summary_path = mlp_root / "ppo_summary.json"
    task_gnn_summary_path = task_gnn_root / "task_gnn_summary.json"
    mlp_manifest_path = mlp_root / "ppo_run_manifest.json"
    task_gnn_manifest_path = task_gnn_root / "task_gnn_run_manifest.json"
    mlp_summary = _load_json(mlp_summary_path)
    task_gnn_summary = _load_json(task_gnn_summary_path)
    mlp_manifest = _load_json(mlp_manifest_path)
    task_gnn_manifest = _load_json(task_gnn_manifest_path)
    if mlp_summary.get("mode") != "stg_masked_ppo":
        raise ValueError("MLP summary has the wrong mode")
    if task_gnn_summary.get("mode") != "stg_task_gnn_ppo":
        raise ValueError("task-GNN summary has the wrong mode")
    for name, value in {
        "MLP summary": mlp_summary,
        "task-GNN summary": task_gnn_summary,
        "MLP manifest": mlp_manifest,
        "task-GNN manifest": task_gnn_manifest,
    }.items():
        if (
            value.get("data_access", value.get("inputs", {})).get("test_accessed")
            is not False
        ):
            raise ValueError(f"{name} does not prove test_accessed=false")

    mlp_seeds = _summary_seed_map(mlp_summary)
    task_gnn_seeds = _summary_seed_map(task_gnn_summary)
    if set(mlp_seeds) != set(task_gnn_seeds) or len(mlp_seeds) < 3:
        raise ValueError("MLP and task-GNN must have the same three or more seeds")

    seed_ids = sorted(mlp_seeds)
    per_seed: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    scenario_ids: list[str] | None = None
    mlp_matrix: list[list[float]] = []
    task_gnn_matrix: list[list[float]] = []
    for seed in seed_ids:
        mlp_diagnostic = mlp_root / f"seed_{seed}_validation_diagnostics.json"
        task_gnn_diagnostic = (
            task_gnn_root / f"seed_{seed}_task_gnn_validation_diagnostics.json"
        )
        mlp_metrics, mlp_rows = _diagnostic_rows(mlp_diagnostic)
        task_gnn_metrics, task_gnn_rows = _diagnostic_rows(task_gnn_diagnostic)
        if set(mlp_rows) != set(task_gnn_rows):
            raise ValueError(f"scenario ids differ for seed {seed}")
        current_ids = sorted(mlp_rows)
        if scenario_ids is None:
            scenario_ids = current_ids
        if current_ids != scenario_ids:
            raise ValueError("scenario ids differ across seeds")
        mlp_values: list[float] = []
        task_gnn_values: list[float] = []
        for scenario_id in current_ids:
            mlp_row = mlp_rows[scenario_id]
            task_gnn_row = task_gnn_rows[scenario_id]
            if mlp_row.get("scenario_hash") != task_gnn_row.get("scenario_hash"):
                raise ValueError(f"scenario hash differs for {scenario_id}")
            mlp_ratio = float(mlp_row["score_ratio"])
            task_gnn_ratio = float(task_gnn_row["score_ratio"])
            delta = task_gnn_ratio - mlp_ratio
            mlp_values.append(mlp_ratio)
            task_gnn_values.append(task_gnn_ratio)
            paired_rows.append(
                {
                    "seed": seed,
                    "scenario_id": scenario_id,
                    "scenario_hash": mlp_row["scenario_hash"],
                    "masked_mlp_ratio": mlp_ratio,
                    "task_gnn_ratio": task_gnn_ratio,
                    "task_gnn_minus_mlp": delta,
                    "outcome": _outcome(delta),
                    "masked_mlp_status": mlp_row["status"],
                    "task_gnn_status": task_gnn_row["status"],
                }
            )
        mlp_array = np.asarray(mlp_values, dtype=np.float64)
        task_gnn_array = np.asarray(task_gnn_values, dtype=np.float64)
        delta_array = task_gnn_array - mlp_array
        mlp_matrix.append(mlp_values)
        task_gnn_matrix.append(task_gnn_values)
        per_seed.append(
            {
                "seed": seed,
                "masked_mlp_mean_ratio": float(np.mean(mlp_array)),
                "task_gnn_mean_ratio": float(np.mean(task_gnn_array)),
                "mean_paired_delta": float(np.mean(delta_array)),
                "masked_mlp_p50_ratio": float(np.percentile(mlp_array, 50)),
                "task_gnn_p50_ratio": float(np.percentile(task_gnn_array, 50)),
                "masked_mlp_p95_ratio": float(np.percentile(mlp_array, 95)),
                "task_gnn_p95_ratio": float(np.percentile(task_gnn_array, 95)),
                "pairs": _counts(delta_array),
                "masked_mlp_failure_count": int(mlp_metrics["failure_count"]),
                "task_gnn_failure_count": int(task_gnn_metrics["failure_count"]),
                "masked_mlp_illegal_action_count": int(
                    mlp_metrics["illegal_action_count"]
                ),
                "task_gnn_illegal_action_count": int(
                    task_gnn_metrics["illegal_action_count"]
                ),
            }
        )

    mlp_array = np.asarray(mlp_matrix, dtype=np.float64)
    task_gnn_array = np.asarray(task_gnn_matrix, dtype=np.float64)
    paired = _paired_statistics(
        mlp_array,
        task_gnn_array,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    assert scenario_ids is not None
    scenario_comparison = []
    for scenario_index, scenario_id in enumerate(scenario_ids):
        mlp_mean = float(np.mean(mlp_array[:, scenario_index]))
        task_gnn_mean = float(np.mean(task_gnn_array[:, scenario_index]))
        delta = task_gnn_mean - mlp_mean
        scenario_comparison.append(
            {
                "scenario_id": scenario_id,
                "masked_mlp_seed_mean_ratio": mlp_mean,
                "task_gnn_seed_mean_ratio": task_gnn_mean,
                "task_gnn_minus_mlp": delta,
                "outcome": _outcome(delta),
            }
        )

    mlp_actors = []
    task_gnn_actors = []
    for seed in seed_ids:
        mlp_path = mlp_root / mlp_seeds[seed]["best_checkpoint"]["actor"]["name"]
        task_gnn_path = (
            task_gnn_root / task_gnn_seeds[seed]["best_checkpoint"]["actor"]["name"]
        )
        mlp = MaskedMLPPolicy.load(mlp_path)
        task_gnn = TaskGNNPolicy.load(task_gnn_path)
        mlp_actors.append(
            {
                "seed": seed,
                "file": mlp_path.name,
                "file_sha256": _file_hash(mlp_path),
                "parameter_sha256": policy_parameter_hash(mlp),
                "parameter_count": sum(
                    int(value.size) for value in mlp.params.values()
                ),
            }
        )
        task_gnn_actors.append(
            {
                "seed": seed,
                "file": task_gnn_path.name,
                "file_sha256": _file_hash(task_gnn_path),
                "parameter_sha256": task_gnn_parameter_hash(task_gnn),
                "parameter_count": task_gnn.parameter_count,
            }
        )

    latency = None
    if args.latency_repeats > 0:
        if args.benchmark_manifest is None or args.raw_root is None:
            raise ValueError(
                "latency measurement requires benchmark manifest and raw root"
            )
        scenarios = load_frozen_split(
            args.raw_root,
            args.benchmark_manifest,
            "validation",
            purpose="evaluation",
        )
        latency = _benchmark_checkpoints(
            mlp_root,
            task_gnn_root,
            mlp_seeds,
            task_gnn_seeds,
            scenarios,
            args.latency_repeats,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)

    seed_mean_delta = np.mean(task_gnn_array, axis=1) - np.mean(
        mlp_array,
        axis=1,
    )
    evidence_gate = (
        paired["hierarchical_paired_bootstrap"]["excludes_zero_in_task_gnn_direction"]
        and _counts(seed_mean_delta)["task_gnn_win"] >= 2
        and all(
            item["task_gnn_failure_count"] == 0
            and item["task_gnn_illegal_action_count"] == 0
            for item in per_seed
        )
    )
    report = {
        "format_version": 1,
        "mode": "paired_validation_development_comparison",
        "inputs": {
            "mlp_summary": {
                "name": mlp_summary_path.name,
                "sha256": _file_hash(mlp_summary_path),
            },
            "task_gnn_summary": {
                "name": task_gnn_summary_path.name,
                "sha256": _file_hash(task_gnn_summary_path),
            },
            "mlp_run_manifest": {
                "name": mlp_manifest_path.name,
                "sha256": _file_hash(mlp_manifest_path),
                "code": mlp_manifest.get("code"),
            },
            "task_gnn_run_manifest": {
                "name": task_gnn_manifest_path.name,
                "sha256": _file_hash(task_gnn_manifest_path),
                "code": task_gnn_manifest.get("code"),
            },
            "seeds": seed_ids,
            "split": "validation",
            "test_accessed": False,
        },
        "parameters": {
            "masked_mlp": mlp_actors,
            "task_gnn": task_gnn_actors,
            "task_gnn_over_mlp": (
                task_gnn_actors[0]["parameter_count"] / mlp_actors[0]["parameter_count"]
            ),
        },
        "paired_validation": paired,
        "per_seed": per_seed,
        "per_scenario_seed_mean": scenario_comparison,
        "training_wall_seconds": {
            "masked_mlp": args.mlp_wall_seconds,
            "task_gnn": args.task_gnn_wall_seconds,
            "task_gnn_over_mlp": (
                args.task_gnn_wall_seconds / args.mlp_wall_seconds
                if args.mlp_wall_seconds and args.task_gnn_wall_seconds
                else None
            ),
            "boundary": (
                "end-to-end CLI wall times from separate frozen runs; MLP "
                "pipeline includes its feature-ablation reference"
            ),
        },
        "cpu_latency": latency,
        "per_instance_csv": {
            "name": csv_path.name,
            "bytes": csv_path.stat().st_size,
            "sha256": _file_hash(csv_path),
            "row_count": len(paired_rows),
        },
        "development_gate_passed": bool(evidence_gate),
        "recommendation": (
            "retain_task_gnn_as_validation_candidate_for_independent_review"
            if evidence_gate
            else "retain_masked_mlp_and_close_task_gnn_iteration"
        ),
        "claim_boundary": (
            "validation-only paired development evidence; public test and OOD "
            "remain unaccessed, so this is not a final generalization claim"
        ),
    }
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare frozen MLP and task-GNN validation evidence"
    )
    parser.add_argument("--mlp-root", required=True)
    parser.add_argument("--task-gnn-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchmark-manifest", default=None)
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260717)
    parser.add_argument("--latency-repeats", type=int, default=0)
    parser.add_argument("--mlp-wall-seconds", type=float, default=None)
    parser.add_argument("--task-gnn-wall-seconds", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = compare(args)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
