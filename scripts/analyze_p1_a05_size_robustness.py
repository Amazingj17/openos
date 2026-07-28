from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from trisched.env import HeterogeneousDagEnv
from trisched.learning import (
    FEATURE_NAMES,
    TEACHER_FEATURE_NAMES,
    build_candidate_feature_context,
    candidate_features,
)
from trisched.ood import load_materialized_development_slices
from trisched.policies import HeftPolicy, compute_upward_ranks
from trisched.scenario import Scenario


PRIMARY_POLICY = "masked_mlp"
REFERENCE_POLICY = "heft"
ANALYZED_SLICES = ("id_validation", "ood_size")
SELECTED_FEATURES = tuple(
    name for name in FEATURE_NAMES if name not in TEACHER_FEATURE_NAMES
)
SELECTED_FEATURE_INDICES = np.asarray(
    [FEATURE_NAMES.index(name) for name in SELECTED_FEATURES],
    dtype=np.int64,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalized_lf_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _number_summary(values: Sequence[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("diagnostic values must be a non-empty finite vector")
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def _longest_path_task_count(scenario: Scenario) -> int:
    predecessors = scenario.predecessors()
    successors = scenario.successors()
    indegree = [len(items) for items in predecessors]
    ready = [index for index, degree in enumerate(indegree) if degree == 0]
    depth = [1] * scenario.task_count
    visited = 0
    while ready:
        task_id = min(ready)
        ready.remove(task_id)
        visited += 1
        for child in successors[task_id]:
            depth[child] = max(depth[child], depth[task_id] + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if visited != scenario.task_count:
        raise ValueError(f"scenario is cyclic: {scenario.id}")
    return max(depth)


def _scenario_descriptor(scenario: Scenario) -> dict[str, float]:
    predecessors = scenario.predecessors()
    successors = scenario.successors()
    workloads = np.asarray([task.workload for task in scenario.tasks])
    speeds = np.asarray([resource.speed for resource in scenario.resources])
    off_diagonal_bandwidth = np.asarray(
        [
            scenario.bandwidth[source][target]
            for source in range(scenario.resource_count)
            for target in range(scenario.resource_count)
            if source != target
        ],
        dtype=np.float64,
    )
    off_diagonal_latency = np.asarray(
        [
            scenario.latency[source][target]
            for source in range(scenario.resource_count)
            for target in range(scenario.resource_count)
            if source != target
        ],
        dtype=np.float64,
    )
    ranks = compute_upward_ranks(scenario)
    context = build_candidate_feature_context(scenario, ranks)
    possible_edges = scenario.task_count * (scenario.task_count - 1) / 2
    return {
        "task_count": float(scenario.task_count),
        "edge_count": float(len(scenario.edges)),
        "edge_density": float(len(scenario.edges) / possible_edges),
        "root_count": float(sum(not items for items in predecessors)),
        "leaf_count": float(sum(not items for items in successors)),
        "longest_path_task_count": float(_longest_path_task_count(scenario)),
        "max_indegree": float(max(len(items) for items in predecessors)),
        "max_outdegree": float(max(len(items) for items in successors)),
        "workload_mean": float(np.mean(workloads)),
        "workload_std": float(np.std(workloads)),
        "workload_max": float(np.max(workloads)),
        "speed_min": float(np.min(speeds)),
        "speed_max": float(np.max(speeds)),
        "speed_ratio": float(np.max(speeds) / np.min(speeds)),
        "off_diagonal_bandwidth_min": float(np.min(off_diagonal_bandwidth)),
        "off_diagonal_bandwidth_median": float(np.median(off_diagonal_bandwidth)),
        "off_diagonal_bandwidth_max": float(np.max(off_diagonal_bandwidth)),
        "off_diagonal_latency_mean": float(np.mean(off_diagonal_latency)),
        "off_diagonal_latency_max": float(np.max(off_diagonal_latency)),
        "normalization_time_scale": float(context.time_scale),
        "normalization_max_rank": float(context.max_rank),
        "normalization_max_degree": float(context.max_degree),
        "progress_increment": float(1.0 / scenario.task_count),
    }


def _heft_trace_features(scenario: Scenario) -> tuple[np.ndarray, np.ndarray]:
    env = HeterogeneousDagEnv(scenario)
    policy = HeftPolicy()
    policy.reset(scenario)
    ranks = compute_upward_ranks(scenario)
    context = build_candidate_feature_context(scenario, ranks)
    feature_blocks: list[np.ndarray] = []
    candidate_counts: list[int] = []
    sample_indices = {
        0,
        scenario.task_count // 4,
        scenario.task_count // 2,
        3 * scenario.task_count // 4,
        scenario.task_count - 1,
    }
    decision_index = 0
    while not env.done:
        candidate_counts.append(len(env.ready_tasks()) * scenario.resource_count)
        if decision_index in sample_indices:
            _, features = candidate_features(env, ranks, context)
            feature_blocks.append(features[:, SELECTED_FEATURE_INDICES])
        env.step(*policy.select_action(env))
        decision_index += 1
    return (
        np.concatenate(feature_blocks, axis=0),
        np.asarray(candidate_counts, dtype=np.float64),
    )


def _slice_diagnostics(
    scenarios: Sequence[Scenario],
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, dict[str, float]]]:
    descriptors = {
        scenario.id: _scenario_descriptor(scenario) for scenario in scenarios
    }
    descriptor_names = tuple(next(iter(descriptors.values())))
    descriptor_summary = {
        name: _number_summary([item[name] for item in descriptors.values()])
        for name in descriptor_names
    }
    feature_blocks: list[np.ndarray] = []
    candidate_counts: list[np.ndarray] = []
    for scenario in scenarios:
        features, counts = _heft_trace_features(scenario)
        feature_blocks.append(features)
        candidate_counts.append(counts)
    feature_matrix = np.concatenate(feature_blocks, axis=0)
    feature_values = {
        name: feature_matrix[:, index] for index, name in enumerate(SELECTED_FEATURES)
    }
    return (
        {
            "scenario_count": len(scenarios),
            "descriptors": descriptor_summary,
            "heft_trace_candidate_count": _number_summary(
                np.concatenate(candidate_counts)
            ),
            "heft_trace_sampled_features": {
                name: _number_summary(values) for name, values in feature_values.items()
            },
            "heft_trace_feature_sample_points_per_scenario": 5,
        },
        feature_values,
        descriptors,
    )


def _policy_performance(
    records: Sequence[Mapping[str, Any]],
    *,
    slice_id: str,
    policy: str,
    tie_tolerance: float,
) -> dict[str, Any]:
    selected = [
        item
        for item in records
        if item.get("slice_id") == slice_id and item.get("policy") == policy
    ]
    if not selected or any(item.get("status") != "success" for item in selected):
        raise ValueError(f"missing or failed records for {slice_id}/{policy}")
    ratios = np.asarray([float(item["ratio"]) for item in selected])
    per_seed: dict[int, list[float]] = defaultdict(list)
    per_scenario: dict[str, list[float]] = defaultdict(list)
    for item in selected:
        per_seed[int(item["seed"])].append(float(item["ratio"]))
        per_scenario[str(item["scenario_id"])].append(float(item["ratio"]))
    return {
        "record_count": len(selected),
        "ratio": _number_summary(ratios),
        "win": int(np.count_nonzero(ratios < 1.0 - tie_tolerance)),
        "tie": int(np.count_nonzero(np.abs(ratios - 1.0) <= tie_tolerance)),
        "loss": int(np.count_nonzero(ratios > 1.0 + tie_tolerance)),
        "per_seed_mean": {
            str(seed): float(np.mean(values))
            for seed, values in sorted(per_seed.items())
        },
        "per_scenario_mean": {
            scenario_id: float(np.mean(values))
            for scenario_id, values in sorted(per_scenario.items())
        },
    }


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size != y.size or x.size < 2:
        raise ValueError("correlation vectors must have equal non-trivial length")
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def analyze(
    *,
    contract_path: Path,
    materialization_root: Path,
    evidence_path: Path,
    training_summary_path: Path,
) -> dict[str, Any]:
    contract = _load_json(contract_path)
    evidence = _load_json(evidence_path)
    training = _load_json(training_summary_path)
    if contract.get("primary_policy") != PRIMARY_POLICY:
        raise ValueError("unexpected primary policy")
    if contract.get("reference_policy") != REFERENCE_POLICY:
        raise ValueError("unexpected reference policy")
    development_slices = contract.get("modes", {}).get("development")
    if not isinstance(development_slices, list) or "public_test" in development_slices:
        raise ValueError("development mode must exclude public test")
    if (
        evidence.get("mode") != "development"
        or evidence.get("test_accessed") is not False
    ):
        raise ValueError("diagnostic evidence must be development-only")
    producer = evidence.get("producer")
    if not isinstance(producer, dict) or any(
        producer.get(name) is not False
        for name in ("training_started", "public_test_loaded")
    ):
        raise ValueError("diagnostic evidence crossed a training/test boundary")
    access = training.get("data_access")
    if not isinstance(access, dict) or access.get("test_accessed") is not False:
        raise ValueError("training summary must prove test was not accessed")

    slices, manifest = load_materialized_development_slices(
        contract_path,
        materialization_root,
    )
    if manifest.get("test_accessed") is not False:
        raise ValueError("materialization manifest crossed the test boundary")
    if any(name not in slices for name in ANALYZED_SLICES):
        raise ValueError("required diagnostic slice is missing")
    records = evidence.get("records")
    if not isinstance(records, list) or any(
        not isinstance(item, dict) for item in records
    ):
        raise ValueError("evidence records must be an array of objects")
    if any(item.get("slice_id") == "public_test" for item in records):
        raise ValueError("public-test evidence is forbidden")

    slice_reports: dict[str, Any] = {}
    feature_values: dict[str, dict[str, np.ndarray]] = {}
    descriptors: dict[str, dict[str, dict[str, float]]] = {}
    for slice_id in ANALYZED_SLICES:
        report, values, raw_descriptors = _slice_diagnostics(slices[slice_id])
        slice_reports[slice_id] = report
        feature_values[slice_id] = values
        descriptors[slice_id] = raw_descriptors

    feature_support_shift: dict[str, dict[str, float]] = {}
    for name in SELECTED_FEATURES:
        baseline = feature_values["id_validation"][name]
        shifted = feature_values["ood_size"][name]
        baseline_min = float(np.min(baseline))
        baseline_max = float(np.max(baseline))
        baseline_std = float(np.std(baseline))
        feature_support_shift[name] = {
            "size_outside_id_min_max_fraction": float(
                np.mean((shifted < baseline_min) | (shifted > baseline_max))
            ),
            "mean_shift_in_id_std": (
                float((np.mean(shifted) - np.mean(baseline)) / baseline_std)
                if baseline_std > 0.0
                else 0.0
            ),
        }

    tie_tolerance = float(contract["tie_tolerance"])
    performance = {
        slice_id: {
            policy: _policy_performance(
                records,
                slice_id=slice_id,
                policy=policy,
                tie_tolerance=tie_tolerance,
            )
            for policy in (REFERENCE_POLICY, "bc", PRIMARY_POLICY, "task_gnn")
        }
        for slice_id in ANALYZED_SLICES
    }
    size_primary = performance["ood_size"][PRIMARY_POLICY]
    size_scenario_ratios = size_primary["per_scenario_mean"]
    size_descriptor_correlations: dict[str, float | None] = {}
    descriptor_names = tuple(next(iter(descriptors["ood_size"].values())))
    scenario_ids = sorted(size_scenario_ratios)
    for name in descriptor_names:
        size_descriptor_correlations[name] = _pearson(
            [
                descriptors["ood_size"][scenario_id][name]
                for scenario_id in scenario_ids
            ],
            [size_scenario_ratios[scenario_id] for scenario_id in scenario_ids],
        )

    teacher = training.get("teacher")
    training_seeds = training.get("seeds")
    if not isinstance(teacher, dict) or not isinstance(training_seeds, list):
        raise ValueError("training summary is missing teacher/seed evidence")
    train_scenarios = int(teacher["scenario_count"])
    train_actions = int(teacher["action_count"])
    if train_scenarios <= 0 or train_actions % train_scenarios != 0:
        raise ValueError("training teacher horizon is not reconstructible")
    fixed_train_task_count = train_actions // train_scenarios
    selected_warm_start_seeds = sorted(
        int(item["seed"])
        for item in training_seeds
        if item.get("selected_warm_start") is True
    )
    warm_start_size_means = {
        str(seed): size_primary["per_seed_mean"][str(seed)]
        for seed in selected_warm_start_seeds
    }

    return {
        "format_version": 1,
        "task": "P1-A05-SIZE-ROBUSTNESS-DESIGN",
        "inputs": {
            "contract": {
                "path": contract_path.resolve().relative_to(REPOSITORY).as_posix(),
                "canonical_sha256": _canonical_json_sha256(contract),
            },
            "materialization_manifest_sha256": producer[
                "materialization_manifest_sha256"
            ],
            "evidence": {
                "path": evidence_path.resolve().relative_to(REPOSITORY).as_posix(),
                "sha256": _file_sha256(evidence_path),
                "records_sha256": evidence["records_sha256"],
            },
            "training_summary": {
                "path": training_summary_path.resolve()
                .relative_to(REPOSITORY)
                .as_posix(),
                "sha256": _file_sha256(training_summary_path),
            },
            "analysis_script_normalized_lf_sha256": _normalized_lf_sha256(
                Path(__file__)
            ),
        },
        "boundaries": {
            "test_accessed": False,
            "public_test_loaded": False,
            "training_started": False,
            "analyzed_slices": list(ANALYZED_SLICES),
        },
        "performance": performance,
        "scenario_and_trace_distribution": slice_reports,
        "feature_support_shift": feature_support_shift,
        "size_descriptor_pearson_correlation": size_descriptor_correlations,
        "mechanism_checks": {
            "fixed_train_task_count": fixed_train_task_count,
            "size_task_count": int(
                slice_reports["ood_size"]["descriptors"]["task_count"]["mean"]
            ),
            "horizon_multiplier": float(
                slice_reports["ood_size"]["descriptors"]["task_count"]["mean"]
                / fixed_train_task_count
            ),
            "selected_warm_start_seeds": selected_warm_start_seeds,
            "selected_warm_start_size_mean_ratio": warm_start_size_means,
            "all_masked_mlp_size_records_lose_to_heft": (
                size_primary["loss"] == size_primary["record_count"]
            ),
            "all_task_gnn_size_records_lose_to_heft": (
                performance["ood_size"]["task_gnn"]["loss"]
                == performance["ood_size"]["task_gnn"]["record_count"]
            ),
        },
        "interpretation_limits": [
            "ood_size changes task count and generator/domain together; it is not a pure size intervention",
            "correlations are descriptive over 30 frozen scenarios and are not causal estimates",
            "no new policy was trained or selected from these diagnostics",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build read-only P1-A05 size-robustness diagnostics",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=REPOSITORY / "configs" / "p1_b02_evaluation_contract.json",
    )
    parser.add_argument(
        "--materialization-root",
        type=Path,
        default=REPOSITORY / "outputs" / "p1-b02-development-slices-v2",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=REPOSITORY
        / "outputs"
        / "p1-b02-development-evidence"
        / "development-evidence.json",
    )
    parser.add_argument(
        "--training-summary",
        type=Path,
        default=REPOSITORY / "outputs" / "p1-a04-stg-ppo-5seed" / "ppo_summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY
        / "outputs"
        / "p1-a05-size-robustness-design"
        / "root_cause_analysis.json",
    )
    args = parser.parse_args(argv)
    report = analyze(
        contract_path=args.contract,
        materialization_root=args.materialization_root,
        evidence_path=args.evidence,
        training_summary_path=args.training_summary,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
