from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.compare_p1_a05_development import (
    _paired_statistics,
    build_comparison,
)
from trisched.hashing import canonical_json_sha256, file_sha256
from trisched.reporting import load_evaluation_contract, scenario_set_sha256


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs" / "p1_b02_evaluation_contract.json"


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _scenario_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _evidence(
    contract: dict[str, Any],
    *,
    commit: str,
    candidate: bool,
) -> dict[str, Any]:
    primary = contract["primary_policy"]
    records: list[dict[str, Any]] = []
    manifests: dict[str, Any] = {}
    baseline_means = {
        "id_validation": 0.70,
        "ood_size": 1.50,
        "ood_ccr": 0.40,
        "ood_system": 0.96,
    }
    candidate_means = {
        "id_validation": 0.71,
        "ood_size": 0.80,
        "ood_ccr": 0.45,
        "ood_system": 0.90,
    }
    slices = {
        item["id"]: item
        for item in contract["slices"]
        if item["id"] in contract["modes"]["development"]
    }
    for slice_id in contract["modes"]["development"]:
        scenarios = [
            {
                "scenario_id": f"{slice_id}-{index:04d}",
                "scenario_hash": _scenario_hash(f"{slice_id}-{index:04d}"),
            }
            for index in range(int(slices[slice_id]["scenario_count"]))
        ]
        manifests[slice_id] = {
            "scenario_count": len(scenarios),
            "scenario_set_sha256": scenario_set_sha256(scenarios),
        }
        for policy in contract["policies"]:
            for seed_index, seed in enumerate(policy["required_seeds"]):
                for scenario_index, scenario in enumerate(scenarios):
                    if policy["id"] == contract["reference_policy"]:
                        ratio = 1.0
                    elif policy["id"] == primary:
                        center = (
                            candidate_means[slice_id]
                            if candidate
                            else baseline_means[slice_id]
                        )
                        ratio = center + seed_index * 0.001 + scenario_index * 0.00001
                    else:
                        ratio = 1.10 + seed_index * 0.001
                    records.append(
                        {
                            "slice_id": slice_id,
                            "policy": policy["id"],
                            "seed": seed,
                            **scenario,
                            "status": "success",
                            "ratio": ratio,
                            "score_ratio": ratio,
                            "penalty_applied": False,
                            "illegal_action_count": 0,
                            "error_code": None,
                            "runtime_ms": 1.0,
                        }
                    )
    result = {
        "format_version": 1,
        "mode": "development",
        "contract_sha256": canonical_json_sha256(contract),
        "code": {
            "commit": commit,
            "working_tree_dirty": False,
            "source": "git",
            "runner_bundle": {"checkpoints": {"masked_mlp": {}}},
        },
        "test_accessed": False,
        "slice_manifests": manifests,
        "records": records,
        "records_sha256": canonical_json_sha256(records),
    }
    return result


def _fixture(tmp_path: Path) -> dict[str, Path | str]:
    contract = load_evaluation_contract(CONTRACT_PATH)
    commit = "a" * 40
    baseline = _evidence(contract, commit="b" * 40, candidate=False)
    candidate = _evidence(contract, commit=commit, candidate=True)
    seeds = next(
        item["required_seeds"]
        for item in contract["policies"]
        if item["id"] == contract["primary_policy"]
    )
    training_checkpoints: dict[str, Any] = {}
    candidate_checkpoints = candidate["code"]["runner_bundle"]["checkpoints"][
        "masked_mlp"
    ]
    for seed in seeds:
        path = tmp_path / "outputs" / "candidate" / f"seed_{seed}_ppo_best_policy.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"checkpoint-{seed}".encode("ascii"))
        parameter_sha256 = _scenario_hash(f"parameters-{seed}")
        relative = path.relative_to(tmp_path).as_posix()
        candidate_checkpoints[str(seed)] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "parameter_sha256": parameter_sha256,
            "internal_seed": seed,
            "feature_names": ["teacher-free"],
        }
        training_checkpoints[str(seed)] = {
            "best": {
                "actor": {
                    "name": path.name,
                    "sha256": file_sha256(path),
                    "parameter_sha256": parameter_sha256,
                }
            }
        }

    baseline_path = _write_json(tmp_path / "baseline.json", baseline)
    candidate["records_sha256"] = canonical_json_sha256(candidate["records"])
    candidate_path = _write_json(tmp_path / "candidate.json", candidate)
    preregister_path = _write_json(tmp_path / "preregister.json", {"test": False})
    review_path = _write_json(tmp_path / "review.json", {"approved": True})
    review_sha256 = file_sha256(review_path)
    summary = {
        "mode": "p1_a05_size_robustness",
        "task_id": "P1-A05-SIZE-ROBUSTNESS",
        "formal_run_count": 1,
        "validation_gate_passed": True,
        "data_access": {"test_accessed": False, "public_test": "forbidden"},
        "implementation_review": {"sha256": review_sha256},
    }
    summary_path = _write_json(tmp_path / "p1_a05_summary.json", summary)
    manifest = {
        "mode": "p1_a05_size_robustness",
        "code": {"commit": commit, "working_tree_dirty": False},
        "execution": {"formal_run_count": 1},
        "inputs": {
            "test_accessed": False,
            "public_test": "forbidden",
            "implementation_review_sha256": review_sha256,
        },
        "checkpoints": training_checkpoints,
        "artifacts": {
            summary_path.name: {
                "bytes": summary_path.stat().st_size,
                "sha256": file_sha256(summary_path),
            }
        },
    }
    manifest_path = _write_json(tmp_path / "p1_a05_run_manifest.json", manifest)
    candidate_report = {
        "report_scope": "development",
        "evidence": {
            "sha256": file_sha256(candidate_path),
            "records_sha256": candidate["records_sha256"],
        },
        "gate": {
            "primary_zero_failures_and_illegal_actions": True,
            "primary_mean_ratio_below_reference_on_every_reported_slice": True,
            "release_publishable": False,
        },
    }
    report_path = _write_json(tmp_path / "evaluation_report.json", candidate_report)
    return {
        "contract": CONTRACT_PATH,
        "preregister": preregister_path,
        "baseline": baseline_path,
        "candidate": candidate_path,
        "report": report_path,
        "summary": summary_path,
        "manifest": manifest_path,
        "review": review_path,
        "output": tmp_path / "comparison.json",
        "commit": commit,
    }


def _build(values: dict[str, Path | str], tmp_path: Path) -> Path:
    return build_comparison(
        contract_path=Path(values["contract"]),
        preregister_path=Path(values["preregister"]),
        baseline_evidence_path=Path(values["baseline"]),
        candidate_evidence_path=Path(values["candidate"]),
        candidate_report_path=Path(values["report"]),
        training_summary_path=Path(values["summary"]),
        training_manifest_path=Path(values["manifest"]),
        implementation_review_path=Path(values["review"]),
        output_path=Path(values["output"]),
        repository=tmp_path,
        expected_commit=str(values["commit"]),
        enforce_preregister_baseline=False,
    )


def test_paired_statistics_use_shared_seed_and_scenario_resampling() -> None:
    baseline = np.asarray([[1.0, 1.1, 0.9], [1.2, 1.0, 0.8]])
    candidate = baseline - 0.2
    first = _paired_statistics(
        baseline,
        candidate,
        samples=2000,
        seed=17,
        confidence_level=0.95,
        tie_tolerance=1e-9,
    )
    second = _paired_statistics(
        baseline,
        candidate,
        samples=2000,
        seed=17,
        confidence_level=0.95,
        tie_tolerance=1e-9,
    )
    assert first == second
    assert first["pair_count"] == 6
    assert np.isclose(first["mean_paired_delta"], -0.2)
    assert first["all_seed_scenario_pairs"] == {
        "candidate_win": 6,
        "tie": 0,
        "baseline_win": 0,
    }
    assert (
        first["hierarchical_paired_bootstrap"]["supports_candidate_improvement"] is True
    )


def test_comparison_binds_training_evidence_report_and_all_preregistered_gates(
    tmp_path: Path,
) -> None:
    values = _fixture(tmp_path)
    output = _build(values, tmp_path)
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["inputs"]["test_accessed"] is False
    assert report["inputs"]["public_test"] == "forbidden"
    assert report["gates"] == {
        "candidate_zero_failures_and_illegal_actions": True,
        "candidate_mean_ratio_below_heft_on_every_development_slice": True,
        "size_mean_ratio_below_one": True,
        "size_new_minus_p1_a04_ci_upper_below_zero": True,
        "id_mean_ratio_below_one_and_within_baseline_plus_0_02": True,
        "development_gate_passed": True,
    }
    assert report["decision"] == "eligible_for_independent_review_before_G3"
    assert report["paired_development"]["ood_size"]["pair_count"] == 150
    assert report["inputs"]["candidate_standard_report"]["release_publishable"] is False


def test_comparison_rejects_checkpoint_drift_and_public_test_evidence(
    tmp_path: Path,
) -> None:
    values = _fixture(tmp_path)
    manifest_path = Path(values["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first_seed = sorted(manifest["checkpoints"])[0]
    manifest["checkpoints"][first_seed]["best"]["actor"]["sha256"] = "0" * 64
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="checkpoint bytes/parameters differ"):
        _build(values, tmp_path)

    values = _fixture(tmp_path / "public-test")
    candidate_path = Path(values["candidate"])
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["test_accessed"] = True
    _write_json(candidate_path, candidate)
    with pytest.raises(Exception, match="test_accessed"):
        _build(values, tmp_path / "public-test")


def test_comparison_rejects_standard_report_gate_that_disagrees_with_evidence(
    tmp_path: Path,
) -> None:
    values = _fixture(tmp_path)
    report_path = Path(values["report"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["gate"]["primary_mean_ratio_below_reference_on_every_reported_slice"] = False
    _write_json(report_path, report)
    with pytest.raises(ValueError, match="mean-ratio gate differs"):
        _build(values, tmp_path)


def test_comparison_preserves_a_valid_negative_result_and_stops_candidate(
    tmp_path: Path,
) -> None:
    values = _fixture(tmp_path)
    candidate_path = Path(values["candidate"])
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    for row in candidate["records"]:
        if row["policy"] == "masked_mlp" and row["slice_id"] == "ood_size":
            row["ratio"] = 1.2
            row["score_ratio"] = 1.2
    candidate["records_sha256"] = canonical_json_sha256(candidate["records"])
    _write_json(candidate_path, candidate)

    report_path = Path(values["report"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["evidence"] = {
        "sha256": file_sha256(candidate_path),
        "records_sha256": candidate["records_sha256"],
    }
    report["gate"]["primary_mean_ratio_below_reference_on_every_reported_slice"] = False
    _write_json(report_path, report)

    output = _build(values, tmp_path)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["gates"]["development_gate_passed"] is False
    assert result["decision"] == "retain_p1_a04_publish_negative_result_and_stop_p1_a05"


def test_comparison_refuses_to_overwrite_formal_output(tmp_path: Path) -> None:
    values = _fixture(tmp_path)
    output = Path(values["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        _build(values, tmp_path)
