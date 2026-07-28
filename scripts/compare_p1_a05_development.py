from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from trisched.hashing import (
    canonical_json_sha256,
    file_sha256,
    portable_text_hashes,
)
from trisched.reporting import _validate_evidence, load_evaluation_contract


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} cannot be read: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _repository_file(repository: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path must be a non-empty string")
    path = (repository / value).resolve()
    try:
        path.relative_to(repository.resolve())
    except ValueError as error:
        raise ValueError(f"{label} path escapes the repository") from error
    if not path.is_file():
        raise ValueError(f"{label} file does not exist: {path}")
    return path


def _clean_head(repository: Path) -> str:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        text=True,
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repository,
        text=True,
    ).strip()
    if dirty:
        raise ValueError("P1-A05 comparison requires a clean tracked worktree")
    return commit


def _policy_seed_map(contract: Mapping[str, Any], policy_id: str) -> list[int]:
    for item in contract["policies"]:
        if item["id"] == policy_id:
            return list(item["required_seeds"])
    raise ValueError(f"policy is absent from the contract: {policy_id}")


def _paired_statistics(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    samples: int,
    seed: int,
    confidence_level: float,
    tie_tolerance: float,
) -> dict[str, Any]:
    baseline_values = np.asarray(baseline, dtype=np.float64)
    candidate_values = np.asarray(candidate, dtype=np.float64)
    if (
        baseline_values.shape != candidate_values.shape
        or baseline_values.ndim != 2
        or baseline_values.shape[0] < 1
        or baseline_values.shape[1] < 2
    ):
        raise ValueError("paired inputs must have the same seed-by-scenario shape")
    if samples <= 0 or not 0.0 < confidence_level < 1.0:
        raise ValueError("invalid paired bootstrap configuration")
    if not np.all(np.isfinite(baseline_values)) or not np.all(
        np.isfinite(candidate_values)
    ):
        raise ValueError("paired ratios must be finite")

    delta = candidate_values - baseline_values
    rng = np.random.default_rng(seed)
    sampled = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        seed_indices = rng.integers(0, delta.shape[0], size=delta.shape[0])
        scenario_indices = rng.integers(0, delta.shape[1], size=delta.shape[1])
        sampled[index] = float(np.mean(delta[np.ix_(seed_indices, scenario_indices)]))
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.percentile(
        sampled,
        [100.0 * alpha, 100.0 * (1.0 - alpha)],
    )
    scenario_delta = np.mean(delta, axis=0)

    def counts(values: np.ndarray) -> dict[str, int]:
        return {
            "candidate_win": int(np.sum(values < -tie_tolerance)),
            "tie": int(np.sum(np.abs(values) <= tie_tolerance)),
            "baseline_win": int(np.sum(values > tie_tolerance)),
        }

    return {
        "direction": "lower_score_ratio_is_better",
        "delta_definition": "p1_a05_candidate_minus_p1_a04_baseline",
        "seed_count": int(delta.shape[0]),
        "scenario_count": int(delta.shape[1]),
        "pair_count": int(delta.size),
        "baseline_mean_ratio": float(np.mean(baseline_values)),
        "candidate_mean_ratio": float(np.mean(candidate_values)),
        "mean_paired_delta": float(np.mean(delta)),
        "median_paired_delta": float(np.median(delta)),
        "all_seed_scenario_pairs": counts(delta),
        "scenario_seed_mean_pairs": counts(scenario_delta),
        "hierarchical_paired_bootstrap": {
            "samples": samples,
            "seed": seed,
            "confidence_level": confidence_level,
            "resampling": (
                "shared training-seed indices and shared scenario indices with "
                "replacement; mean candidate-minus-baseline score ratio"
            ),
            "delta_lower": float(lower),
            "delta_upper": float(upper),
            "supports_candidate_improvement": bool(upper < -tie_tolerance),
        },
    }


def _score_matrix(
    indexed: Mapping[tuple[str, str, int, str], Mapping[str, Any]],
    *,
    slice_id: str,
    policy_id: str,
    seeds: list[int],
    scenario_ids: list[str],
) -> np.ndarray:
    return np.asarray(
        [
            [
                float(indexed[(slice_id, policy_id, seed, scenario_id)]["score_ratio"])
                for scenario_id in scenario_ids
            ]
            for seed in seeds
        ],
        dtype=np.float64,
    )


def _verify_training_bindings(
    *,
    repository: Path,
    expected_commit: str,
    seeds: list[int],
    candidate_evidence: Mapping[str, Any],
    summary_path: Path,
    manifest_path: Path,
    review_path: Path,
) -> dict[str, Any]:
    summary = _load_object(summary_path, "P1-A05 training summary")
    manifest = _load_object(manifest_path, "P1-A05 run manifest")
    review_sha256 = file_sha256(review_path)
    if (
        summary.get("mode") != "p1_a05_size_robustness"
        or summary.get("task_id") != "P1-A05-SIZE-ROBUSTNESS"
        or summary.get("formal_run_count") != 1
        or summary.get("validation_gate_passed") is not True
        or summary.get("data_access", {}).get("test_accessed") is not False
        or summary.get("data_access", {}).get("public_test") != "forbidden"
        or summary.get("implementation_review", {}).get("sha256") != review_sha256
    ):
        raise ValueError("P1-A05 summary does not satisfy the formal training gates")
    if (
        manifest.get("mode") != "p1_a05_size_robustness"
        or manifest.get("code", {}).get("commit") != expected_commit
        or manifest.get("code", {}).get("working_tree_dirty") is not False
        or manifest.get("execution", {}).get("formal_run_count") != 1
        or manifest.get("inputs", {}).get("test_accessed") is not False
        or manifest.get("inputs", {}).get("public_test") != "forbidden"
        or manifest.get("inputs", {}).get("implementation_review_sha256")
        != review_sha256
    ):
        raise ValueError("P1-A05 run manifest does not bind the formal clean run")
    summary_artifact = manifest.get("artifacts", {}).get(summary_path.name)
    if (
        not isinstance(summary_artifact, dict)
        or summary_artifact.get("bytes") != summary_path.stat().st_size
        or summary_artifact.get("sha256") != file_sha256(summary_path)
    ):
        raise ValueError("P1-A05 summary artifact binding is invalid")

    candidate_checkpoints = (
        candidate_evidence.get("code", {})
        .get("runner_bundle", {})
        .get("checkpoints", {})
        .get("masked_mlp")
    )
    training_checkpoints = manifest.get("checkpoints")
    if not isinstance(candidate_checkpoints, dict) or not isinstance(
        training_checkpoints, dict
    ):
        raise ValueError("candidate/training checkpoint metadata is absent")
    checkpoint_bindings: dict[str, Any] = {}
    for seed in seeds:
        key = str(seed)
        runner_item = candidate_checkpoints.get(key)
        training_item = training_checkpoints.get(key, {}).get("best", {}).get("actor")
        if not isinstance(runner_item, dict) or not isinstance(training_item, dict):
            raise ValueError(f"checkpoint binding is absent for seed {seed}")
        checkpoint_path = _repository_file(
            repository,
            runner_item.get("path"),
            f"candidate checkpoint {seed}",
        )
        if (
            runner_item.get("internal_seed") != seed
            or runner_item.get("sha256") != training_item.get("sha256")
            or runner_item.get("parameter_sha256")
            != training_item.get("parameter_sha256")
            or checkpoint_path.name != training_item.get("name")
            or runner_item.get("bytes") != checkpoint_path.stat().st_size
            or runner_item.get("sha256") != file_sha256(checkpoint_path)
        ):
            raise ValueError(f"checkpoint bytes/parameters differ for seed {seed}")
        checkpoint_bindings[key] = {
            "path": runner_item["path"],
            "bytes": runner_item["bytes"],
            "sha256": runner_item["sha256"],
            "parameter_sha256": runner_item["parameter_sha256"],
        }
    return {
        "summary": {
            "path": summary_path.name,
            "bytes": summary_path.stat().st_size,
            "sha256": file_sha256(summary_path),
        },
        "run_manifest": {
            "path": manifest_path.name,
            "bytes": manifest_path.stat().st_size,
            "sha256": file_sha256(manifest_path),
            "code_commit": expected_commit,
        },
        "implementation_review_sha256": review_sha256,
        "checkpoints": checkpoint_bindings,
    }


def _verify_candidate_report(
    report_path: Path,
    candidate_evidence_path: Path,
    candidate_evidence: Mapping[str, Any],
    *,
    expected_primary_zero_failures: bool,
) -> dict[str, Any]:
    report = _load_object(report_path, "candidate development report")
    gate = report.get("gate")
    if (
        report.get("report_scope") != "development"
        or report.get("evidence", {}).get("sha256")
        != file_sha256(candidate_evidence_path)
        or report.get("evidence", {}).get("records_sha256")
        != candidate_evidence.get("records_sha256")
        or not isinstance(gate, dict)
        or gate.get("primary_zero_failures_and_illegal_actions")
        is not expected_primary_zero_failures
        or gate.get("release_publishable") is not False
    ):
        raise ValueError("candidate standard development report binding is invalid")
    return {
        "path": report_path.name,
        "bytes": report_path.stat().st_size,
        "sha256": file_sha256(report_path),
        "primary_zero_failures_and_illegal_actions": expected_primary_zero_failures,
        "primary_mean_ratio_below_reference_on_every_reported_slice": gate.get(
            "primary_mean_ratio_below_reference_on_every_reported_slice"
        ),
        "release_publishable": gate.get("release_publishable"),
        "release_publishable_boundary": (
            "expected false in development mode; G3 uses the development gates, "
            "while final release remains behind the separate public-test gate"
        ),
    }


def build_comparison(
    *,
    contract_path: Path,
    preregister_path: Path,
    baseline_evidence_path: Path,
    candidate_evidence_path: Path,
    candidate_report_path: Path,
    training_summary_path: Path,
    training_manifest_path: Path,
    implementation_review_path: Path,
    output_path: Path,
    repository: Path = REPOSITORY,
    expected_commit: str | None = None,
    enforce_preregister_baseline: bool = True,
) -> Path:
    paths = [
        contract_path,
        preregister_path,
        baseline_evidence_path,
        candidate_evidence_path,
        candidate_report_path,
        training_summary_path,
        training_manifest_path,
        implementation_review_path,
    ]
    if any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise ValueError(f"required P1-A05 comparison inputs are absent: {missing}")
    if output_path.exists():
        raise ValueError(f"comparison output already exists: {output_path}")
    commit = expected_commit or _clean_head(repository)

    contract = load_evaluation_contract(contract_path)
    contract_hash = canonical_json_sha256(contract)
    preregister = _load_object(preregister_path, "P1-A05 preregistration")
    expected_contract_hash = (
        preregister.get("evidence_bindings", {})
        .get("evaluation_contract", {})
        .get("canonical_sha256")
    )
    if enforce_preregister_baseline and expected_contract_hash != contract_hash:
        raise ValueError("evaluation contract differs from the frozen preregistration")
    if enforce_preregister_baseline:
        expected_baseline_sha256 = (
            preregister.get("evidence_bindings", {})
            .get("accepted_development_evidence", {})
            .get("sha256")
        )
        if file_sha256(baseline_evidence_path) != expected_baseline_sha256:
            raise ValueError(
                "baseline evidence differs from the frozen preregistration"
            )

    baseline, baseline_index, baseline_scenarios, _ = _validate_evidence(
        baseline_evidence_path,
        contract,
        contract_hash,
        None,
    )
    candidate, candidate_index, candidate_scenarios, _ = _validate_evidence(
        candidate_evidence_path,
        contract,
        contract_hash,
        None,
    )
    if (
        candidate.get("code", {}).get("commit") != commit
        or candidate.get("code", {}).get("working_tree_dirty") is not False
    ):
        raise ValueError(
            "candidate evidence is not bound to the clean comparison commit"
        )
    if baseline_scenarios != candidate_scenarios:
        raise ValueError("baseline and candidate scenario IDs/hashes differ")

    policy_id = contract["primary_policy"]
    seeds = _policy_seed_map(contract, policy_id)
    candidate_primary_rows = [
        row for row in candidate["records"] if row["policy"] == policy_id
    ]
    candidate_primary_zero = all(
        row["status"] == "success" and int(row["illegal_action_count"]) == 0
        for row in candidate_primary_rows
    )
    training_bindings = _verify_training_bindings(
        repository=repository,
        expected_commit=commit,
        seeds=seeds,
        candidate_evidence=candidate,
        summary_path=training_summary_path,
        manifest_path=training_manifest_path,
        review_path=implementation_review_path,
    )
    report_binding = _verify_candidate_report(
        candidate_report_path,
        candidate_evidence_path,
        candidate,
        expected_primary_zero_failures=candidate_primary_zero,
    )

    slice_results: dict[str, Any] = {}
    for slice_id in contract["modes"]["development"]:
        scenario_ids = [item["scenario_id"] for item in baseline_scenarios[slice_id]]
        baseline_matrix = _score_matrix(
            baseline_index,
            slice_id=slice_id,
            policy_id=policy_id,
            seeds=seeds,
            scenario_ids=scenario_ids,
        )
        candidate_matrix = _score_matrix(
            candidate_index,
            slice_id=slice_id,
            policy_id=policy_id,
            seeds=seeds,
            scenario_ids=scenario_ids,
        )
        paired = _paired_statistics(
            baseline_matrix,
            candidate_matrix,
            samples=int(contract["bootstrap"]["samples"]),
            seed=int(contract["bootstrap"]["seed"]),
            confidence_level=float(contract["bootstrap"]["confidence_level"]),
            tie_tolerance=float(contract["tie_tolerance"]),
        )
        paired["baseline_seed_mean_ratios"] = [
            {"seed": seed, "mean_ratio": float(np.mean(baseline_matrix[index]))}
            for index, seed in enumerate(seeds)
        ]
        paired["candidate_seed_mean_ratios"] = [
            {"seed": seed, "mean_ratio": float(np.mean(candidate_matrix[index]))}
            for index, seed in enumerate(seeds)
        ]
        slice_results[slice_id] = paired

    id_result = slice_results["id_validation"]
    size_result = slice_results["ood_size"]
    candidate_below_heft = all(
        result["candidate_mean_ratio"] < 1.0 for result in slice_results.values()
    )
    id_guardrail = bool(
        id_result["candidate_mean_ratio"] < 1.0
        and id_result["candidate_mean_ratio"] <= id_result["baseline_mean_ratio"] + 0.02
    )
    size_effect = bool(
        size_result["candidate_mean_ratio"] < 1.0
        and size_result["hierarchical_paired_bootstrap"][
            "supports_candidate_improvement"
        ]
    )
    standard_gate = bool(
        report_binding["primary_mean_ratio_below_reference_on_every_reported_slice"]
    )
    if standard_gate is not candidate_below_heft:
        raise ValueError(
            "candidate report mean-ratio gate differs from the bound evidence"
        )
    development_gate_passed = bool(
        candidate_primary_zero
        and standard_gate
        and candidate_below_heft
        and id_guardrail
        and size_effect
    )

    result = {
        "format_version": 1,
        "task": "P1-A05-DEVELOPMENT-COMPARISON",
        "mode": "single_formal_candidate_development_gate",
        "code": {
            "commit": commit,
            "working_tree_dirty": False,
            "script": {
                "path": Path(__file__)
                .resolve()
                .relative_to(REPOSITORY.resolve())
                .as_posix(),
                "portable_text": portable_text_hashes(Path(__file__)),
            },
        },
        "inputs": {
            "contract": {
                "path": contract_path.name,
                "canonical_sha256": contract_hash,
            },
            "preregister": {
                "path": preregister_path.name,
                "canonical_sha256": canonical_json_sha256(preregister),
            },
            "baseline_evidence": {
                "path": baseline_evidence_path.name,
                "sha256": file_sha256(baseline_evidence_path),
                "records_sha256": baseline["records_sha256"],
                "role": "accepted P1-A04 masked MLP",
            },
            "candidate_evidence": {
                "path": candidate_evidence_path.name,
                "sha256": file_sha256(candidate_evidence_path),
                "records_sha256": candidate["records_sha256"],
                "role": "single formal P1-A05 masked MLP candidate",
            },
            "candidate_standard_report": report_binding,
            "formal_training": training_bindings,
            "test_accessed": False,
            "public_test": "forbidden",
        },
        "paired_development": slice_results,
        "gates": {
            "candidate_zero_failures_and_illegal_actions": candidate_primary_zero,
            "candidate_mean_ratio_below_heft_on_every_development_slice": (
                candidate_below_heft
            ),
            "size_mean_ratio_below_one": size_result["candidate_mean_ratio"] < 1.0,
            "size_new_minus_p1_a04_ci_upper_below_zero": size_result[
                "hierarchical_paired_bootstrap"
            ]["supports_candidate_improvement"],
            "id_mean_ratio_below_one_and_within_baseline_plus_0_02": id_guardrail,
            "development_gate_passed": development_gate_passed,
        },
        "decision": (
            "eligible_for_independent_review_before_G3"
            if development_gate_passed
            else "retain_p1_a04_publish_negative_result_and_stop_p1_a05"
        ),
        "claim_boundary": (
            "development-only paired evidence; this result never authorizes public-test "
            "access and release_publishable remains a later final-test concept"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the one formal P1-A05 development candidate against the frozen "
            "P1-A04 evidence without public-test access"
        )
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=REPOSITORY / "configs" / "p1_b02_evaluation_contract.json",
    )
    parser.add_argument(
        "--preregister",
        type=Path,
        default=REPOSITORY / "configs" / "p1_a05_size_robustness_preregister.json",
    )
    parser.add_argument(
        "--baseline-evidence",
        type=Path,
        default=REPOSITORY
        / "outputs"
        / "p1-b02-development-evidence"
        / "development-evidence.json",
    )
    parser.add_argument(
        "--candidate-evidence",
        type=Path,
        default=REPOSITORY
        / "outputs"
        / "p1-a05-development-evidence"
        / "development-evidence.json",
    )
    parser.add_argument(
        "--candidate-report",
        type=Path,
        default=REPOSITORY
        / "outputs"
        / "p1-a05-development-report"
        / "evaluation_report.json",
    )
    parser.add_argument(
        "--training-summary",
        type=Path,
        default=REPOSITORY
        / "outputs"
        / "p1-a05-size-robustness"
        / "p1_a05_summary.json",
    )
    parser.add_argument(
        "--training-manifest",
        type=Path,
        default=REPOSITORY
        / "outputs"
        / "p1-a05-size-robustness"
        / "p1_a05_run_manifest.json",
    )
    parser.add_argument(
        "--implementation-review",
        type=Path,
        default=REPOSITORY / "configs" / "p1_a05_implementation_review.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY
        / "outputs"
        / "p1-a05-development-comparison"
        / "comparison.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_comparison(
        contract_path=args.contract.resolve(),
        preregister_path=args.preregister.resolve(),
        baseline_evidence_path=args.baseline_evidence.resolve(),
        candidate_evidence_path=args.candidate_evidence.resolve(),
        candidate_report_path=args.candidate_report.resolve(),
        training_summary_path=args.training_summary.resolve(),
        training_manifest_path=args.training_manifest.resolve(),
        implementation_review_path=args.implementation_review.resolve(),
        output_path=args.output.resolve(),
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
