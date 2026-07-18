from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts.build_p1_06_bundle import (
    P106BundleError,
    build_model_result_bundle,
    prepare_review_request,
    verify_model_result_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
SEEDS = [20260717, 20260718, 20260719, 20260720, 20260721]


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _git_repository(path: Path) -> str:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "p1-06-test@example.invalid"],
        cwd=path,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "P1-06 Test"], cwd=path, check=True)
    (path / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=path, check=True)
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def _fixture(tmp_path: Path) -> dict[str, Any]:
    repository = tmp_path / "repository"
    commit = _git_repository(repository)
    config_dir = repository / "configs"
    contract = json.loads(
        (ROOT / "configs" / "p1_06_bundle_contract.json").read_text(encoding="utf-8")
    )
    contract_path = _write_json(config_dir / "p1_06_bundle_contract.json", contract)
    evaluation_contract = {"format_version": 1, "id": "fixture-evaluation"}
    evaluation_contract_path = _write_json(
        config_dir / "p1_b02_evaluation_contract.json", evaluation_contract
    )
    preregister = {"format_version": 1, "task": "fixture-preregister"}
    preregister_path = _write_json(
        config_dir / "p1_a05_size_robustness_preregister.json", preregister
    )
    implementation_review_path = _write_json(
        config_dir / "p1_a05_implementation_review.json",
        {"format_version": 1, "decision": "approve"},
    )

    training_dir = repository / "outputs" / "p1-a05-size-robustness"
    checkpoints: dict[str, dict[str, Any]] = {}
    formal_checkpoints: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        checkpoint = training_dir / f"seed_{seed}_p1_a05_best_policy.npz"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint-{seed}\n".encode("ascii"))
        parameter_sha256 = hashlib.sha256(f"parameter-{seed}".encode()).hexdigest()
        checkpoints[str(seed)] = {
            "best": {
                "actor": {
                    "name": checkpoint.name,
                    "sha256": _sha256(checkpoint),
                    "parameter_sha256": parameter_sha256,
                }
            }
        }
        formal_checkpoints[str(seed)] = {
            "path": checkpoint.relative_to(repository).as_posix(),
            "bytes": checkpoint.stat().st_size,
            "sha256": _sha256(checkpoint),
            "parameter_sha256": parameter_sha256,
        }
    summary_path = _write_json(
        training_dir / "p1_a05_summary.json",
        {
            "format_version": 1,
            "mode": "p1_a05_size_robustness",
            "formal_run_count": 1,
            "validation_gate_passed": True,
            "data_access": {"test_accessed": False, "public_test": "forbidden"},
        },
    )
    manifest_path = _write_json(
        training_dir / "p1_a05_run_manifest.json",
        {
            "format_version": 1,
            "mode": "p1_a05_size_robustness",
            "code": {"commit": commit, "working_tree_dirty": False},
            "execution": {"formal_run_count": 1},
            "inputs": {"test_accessed": False, "public_test": "forbidden"},
            "checkpoints": checkpoints,
        },
    )

    evidence_dir = repository / "outputs"
    baseline_evidence_path = _write_json(
        evidence_dir / "p1-b02-development-evidence" / "development-evidence.json",
        {
            "format_version": 1,
            "mode": "development",
            "test_accessed": False,
            "producer": {"public_test_loaded": False},
            "records_sha256": hashlib.sha256(b"baseline-records").hexdigest(),
        },
    )
    candidate_evidence_path = _write_json(
        evidence_dir / "p1-a05-development-evidence" / "development-evidence.json",
        {
            "format_version": 1,
            "mode": "development",
            "test_accessed": False,
            "producer": {"public_test_loaded": False},
            "code": {"commit": commit, "working_tree_dirty": False},
            "records_sha256": hashlib.sha256(b"candidate-records").hexdigest(),
        },
    )

    report_dir = evidence_dir / "p1-a05-development-report"
    report_path = _write_json(
        report_dir / "evaluation_report.json",
        {
            "format_version": 1,
            "report_scope": "development",
            "gate": {
                "primary_zero_failures_and_illegal_actions": True,
                "primary_mean_ratio_below_reference_on_every_reported_slice": True,
                "release_publishable": False,
            },
        },
    )
    report_artifacts: dict[str, dict[str, Any]] = {
        report_path.name: {
            "bytes": report_path.stat().st_size,
            "sha256": _sha256(report_path),
        }
    }
    for name in (
        "evaluation_per_seed.csv",
        "evaluation_per_slice.csv",
        "evaluation_primary_comparisons.csv",
    ):
        path = report_dir / name
        path.write_text(f"fixture,{name}\n", encoding="utf-8", newline="\n")
        report_artifacts[name] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    _write_json(
        report_dir / "evaluation_report_manifest.json",
        {
            "format_version": 1,
            "mode": "p1_b02_evaluation_report_manifest",
            "test_accessed": False,
            "artifacts": report_artifacts,
        },
    )

    comparison_path = evidence_dir / "p1-a05-development-comparison.json"
    comparison = {
        "format_version": 1,
        "task": "P1-A05-DEVELOPMENT-COMPARISON",
        "code": {"commit": commit, "working_tree_dirty": False},
        "inputs": {
            "contract": {
                "path": evaluation_contract_path.name,
                "canonical_sha256": _canonical_sha256(evaluation_contract),
            },
            "preregister": {
                "path": preregister_path.name,
                "canonical_sha256": _canonical_sha256(preregister),
            },
            "baseline_evidence": {
                "path": baseline_evidence_path.name,
                "sha256": _sha256(baseline_evidence_path),
                "records_sha256": json.loads(
                    baseline_evidence_path.read_text(encoding="utf-8")
                )["records_sha256"],
            },
            "candidate_evidence": {
                "path": candidate_evidence_path.name,
                "sha256": _sha256(candidate_evidence_path),
                "records_sha256": json.loads(
                    candidate_evidence_path.read_text(encoding="utf-8")
                )["records_sha256"],
            },
            "candidate_standard_report": _record(report_path),
            "formal_training": {
                "summary": _record(summary_path),
                "run_manifest": {
                    **_record(manifest_path),
                    "code_commit": commit,
                },
                "implementation_review_sha256": _sha256(implementation_review_path),
                "checkpoints": formal_checkpoints,
            },
            "test_accessed": False,
            "public_test": "forbidden",
        },
        "gates": {
            "candidate_zero_failures_and_illegal_actions": True,
            "candidate_mean_ratio_below_heft_on_every_development_slice": True,
            "size_mean_ratio_below_one": True,
            "size_new_minus_p1_a04_ci_upper_below_zero": True,
            "id_mean_ratio_below_one_and_within_baseline_plus_0_02": True,
            "development_gate_passed": True,
        },
        "decision": "eligible_for_independent_review_before_G3",
    }
    _write_json(comparison_path, comparison)
    independent_review_path = _write_json(
        evidence_dir / "p1-a05-independent-review" / "p1_a05_independent_review.json",
        {
            "format_version": 1,
            "task": "P1-A05-INDEPENDENT-DEVELOPMENT-REVIEW",
            "reviewer": "B",
            "candidate_commit": commit,
            "code": {
                "commit": commit,
                "working_tree_dirty": False,
                "script": {
                    "path": "scripts/review_p1_a05_development.py",
                    "raw_sha256": "a" * 64,
                    "normalized_lf_sha256": "a" * 64,
                },
            },
            "immutable_remote": {"contains_candidate_commit": True},
            "inputs": {"formal": {"comparison_sha256": _sha256(comparison_path)}},
            "normalized_equivalence": {
                "record_count": 2040,
                "records_canonical_sha256": "d" * 64,
                "report_canonical_sha256": "e" * 64,
                "csvs_canonical_sha256": "f" * 64,
                "comparison_canonical_sha256": "1" * 64,
            },
            "checkpoints": formal_checkpoints,
            "assertions": {
                "immutable_remote_commit_verified": True,
                "five_checkpoint_hashes_recomputed": True,
                "normalized_scheduling_records_equal": True,
                "normalized_reports_equal": True,
                "normalized_csvs_equal": True,
                "paired_comparisons_equal": True,
                "public_test_accessed": False,
            },
            "decision": "approve_before_g3",
        },
    )
    g3_authorization_path = _write_json(
        evidence_dir / "p1-06" / "g3_authorization.json",
        {
            "format_version": 1,
            "task": "G3",
            "candidate_commit": commit,
            "release_commit": commit,
            "comparison_sha256": _sha256(comparison_path),
            "independent_review_sha256": _sha256(independent_review_path),
            "development_gate_passed": True,
            "decision": "approve_p1_a05_as_primary",
            "test_accessed": False,
            "public_test": "forbidden",
            "approvals": [
                {"member": "A", "decision": "approve"},
                {"member": "B", "decision": "approve"},
            ],
        },
    )
    kwargs = {
        "repository": repository,
        "contract_path": contract_path,
        "evaluation_contract_path": evaluation_contract_path,
        "preregister_path": preregister_path,
        "implementation_review_path": implementation_review_path,
        "comparison_path": comparison_path,
        "baseline_evidence_path": baseline_evidence_path,
        "candidate_evidence_path": candidate_evidence_path,
        "candidate_report_dir": report_dir,
        "training_dir": training_dir,
        "independent_review_path": independent_review_path,
        "g3_authorization_path": g3_authorization_path,
    }
    return {
        "repository": repository,
        "contract": contract,
        "comparison_path": comparison_path,
        "checkpoint": training_dir / f"seed_{SEEDS[0]}_p1_a05_best_policy.npz",
        "g3_authorization": g3_authorization_path,
        "kwargs": kwargs,
    }


def _prepare_receipts(fixture: dict[str, Any]) -> tuple[Path, Path, Path]:
    repository = fixture["repository"]
    request_path = repository / "outputs" / "p1-06" / "p1_06_review_request.json"
    prepare_review_request(output_path=request_path, **fixture["kwargs"])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request_sha256 = _canonical_sha256(request)
    receipt_paths = []
    for member in ("A", "B"):
        receipt_paths.append(
            _write_json(
                repository
                / "outputs"
                / "p1-06"
                / f"p1_06_review_{member.lower()}.json",
                {
                    "format_version": 1,
                    "task": "P1-06",
                    "member": member,
                    "decision": "approve",
                    "request_canonical_sha256": request_sha256,
                    "candidate_commit": request["candidate_commit"],
                    "release_commit": request["release_commit"],
                    "public_test_accessed": False,
                    "assertions": {
                        name: True
                        for name in fixture["contract"]["p1_06_review"][
                            "required_assertions"
                        ]
                    },
                },
            )
        )
    return request_path, receipt_paths[0], receipt_paths[1]


def test_p1_06_bundle_is_deterministic_and_self_verifying(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    request, review_a, review_b = _prepare_receipts(fixture)
    first = build_model_result_bundle(
        output_path=fixture["repository"] / "outputs" / "release" / "first.zip",
        review_request_path=request,
        review_receipt_paths=[review_a, review_b],
        **fixture["kwargs"],
    )
    second = build_model_result_bundle(
        output_path=fixture["repository"] / "outputs" / "release" / "second.zip",
        review_request_path=request,
        review_receipt_paths=[review_a, review_b],
        **fixture["kwargs"],
    )

    assert first.read_bytes() == second.read_bytes()
    manifest = verify_model_result_bundle(first)
    assert manifest["bundle_type"] == "trisched_model_results_release"
    assert manifest["archive_policy"]["public_test_raw_bytes_included"] is False
    assert [item["member"] for item in manifest["approvals"]] == ["A", "B"]
    assert (
        len(
            [item for item in manifest["files"] if item["role"] == "primary_checkpoint"]
        )
        == 5
    )


def test_p1_06_refuses_failed_development_gate(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    comparison = json.loads(fixture["comparison_path"].read_text(encoding="utf-8"))
    comparison["gates"]["size_mean_ratio_below_one"] = False
    comparison["decision"] = "retain_p1_a04_publish_negative_result_and_stop_p1_a05"
    _write_json(fixture["comparison_path"], comparison)

    with pytest.raises(P106BundleError, match="does not authorize G3"):
        prepare_review_request(
            output_path=fixture["repository"] / "outputs" / "refused.json",
            **fixture["kwargs"],
        )


def test_p1_06_refuses_duplicate_member_receipts(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    request, review_a, _ = _prepare_receipts(fixture)

    with pytest.raises(P106BundleError, match="duplicate P1-06 review member"):
        build_model_result_bundle(
            output_path=fixture["repository"] / "outputs" / "release" / "bad.zip",
            review_request_path=request,
            review_receipt_paths=[review_a, review_a],
            **fixture["kwargs"],
        )


def test_p1_06_refuses_checkpoint_drift_after_review(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    request, review_a, review_b = _prepare_receipts(fixture)
    fixture["checkpoint"].write_bytes(b"tampered\n")

    with pytest.raises(P106BundleError, match="checkpoint 20260717 bytes or SHA-256"):
        build_model_result_bundle(
            output_path=fixture["repository"] / "outputs" / "release" / "bad.zip",
            review_request_path=request,
            review_receipt_paths=[review_a, review_b],
            **fixture["kwargs"],
        )


def test_p1_06_refuses_g3_without_bound_independent_review(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    authorization = json.loads(fixture["g3_authorization"].read_text(encoding="utf-8"))
    authorization["independent_review_sha256"] = "0" * 64
    _write_json(fixture["g3_authorization"], authorization)

    with pytest.raises(P106BundleError, match="G3 authorization"):
        prepare_review_request(
            output_path=fixture["repository"] / "outputs" / "refused.json",
            **fixture["kwargs"],
        )
