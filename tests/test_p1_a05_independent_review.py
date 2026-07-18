from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts.review_p1_a05_development import (
    P1A05ReviewError,
    build_independent_review,
)


SEEDS = [20260717, 20260718, 20260719, 20260720, 20260721]
SLICES = ["id_validation", "ood_size", "ood_ccr", "ood_system"]


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


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


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _git_repository(path: Path) -> str:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "review-test@example.invalid"],
        cwd=path,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Review Test"], cwd=path, check=True)
    (path / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=path, check=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", commit],
        cwd=path,
        check=True,
    )
    return commit


def _evidence(commit: str, contract_hash: str, runtime_offset: float) -> dict[str, Any]:
    records = []
    for slice_index, slice_id in enumerate(SLICES):
        scenario_id = f"{slice_id}-scenario"
        for policy, seeds in (("heft", [0]), ("masked_mlp", SEEDS)):
            for seed_index, seed in enumerate(seeds):
                ratio = 1.0 if policy == "heft" else 0.7 + 0.01 * seed_index
                records.append(
                    {
                        "slice_id": slice_id,
                        "policy": policy,
                        "seed": seed,
                        "scenario_id": scenario_id,
                        "scenario_hash": hashlib.sha256(
                            scenario_id.encode("utf-8")
                        ).hexdigest(),
                        "status": "success",
                        "ratio": ratio,
                        "score_ratio": ratio,
                        "penalty_applied": False,
                        "illegal_action_count": 0,
                        "error_code": None,
                        "runtime_ms": runtime_offset + slice_index + seed_index / 10.0,
                    }
                )
    return {
        "format_version": 1,
        "mode": "development",
        "contract_sha256": contract_hash,
        "code": {"commit": commit, "working_tree_dirty": False},
        "test_accessed": False,
        "producer": {"training_started": False, "public_test_loaded": False},
        "records": records,
        "records_sha256": _canonical_sha256(records),
    }


def _report(
    evidence_path: Path, evidence: dict[str, Any], runtime_offset: float
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "report_scope": "development",
        "evidence": {
            "sha256": _file_sha256(evidence_path),
            "records_sha256": evidence["records_sha256"],
        },
        "slices": [
            {
                "slice_id": slice_id,
                "policies": {
                    "heft": {"mean_ratio": 1.0, "runtime_ms": {"p50": runtime_offset}},
                    "masked_mlp": {
                        "mean_ratio": 0.72,
                        "runtime_ms": {"p50": runtime_offset + 1.0},
                    },
                },
            }
            for slice_id in SLICES
        ],
        "gate": {
            "primary_zero_failures_and_illegal_actions": True,
            "primary_mean_ratio_below_reference_on_every_reported_slice": True,
            "release_publishable": False,
        },
    }


def _write_report(
    report_dir: Path,
    evidence_path: Path,
    evidence: dict[str, Any],
    runtime_offset: float,
) -> Path:
    report_path = _write_json(
        report_dir / "evaluation_report.json",
        _report(evidence_path, evidence, runtime_offset),
    )
    csv_payloads = {
        "evaluation_per_seed.csv": (
            "slice_id,policy,mean_ratio,runtime_mean_ms\n"
            f"id_validation,masked_mlp,0.72,{runtime_offset}\n"
        ),
        "evaluation_per_slice.csv": (
            "slice_id,policy,mean_ratio,runtime_p50_ms\n"
            f"id_validation,masked_mlp,0.72,{runtime_offset}\n"
        ),
        "evaluation_primary_comparisons.csv": (
            "slice_id,primary_policy,mean_paired_delta\n"
            "id_validation,masked_mlp,-0.28\n"
        ),
    }
    artifacts: dict[str, dict[str, Any]] = {
        report_path.name: {
            "bytes": report_path.stat().st_size,
            "sha256": _file_sha256(report_path),
        }
    }
    for name, payload in csv_payloads.items():
        path = report_dir / name
        path.write_text(payload, encoding="utf-8", newline="\n")
        artifacts[name] = {"bytes": path.stat().st_size, "sha256": _file_sha256(path)}
    _write_json(
        report_dir / "evaluation_report_manifest.json",
        {"format_version": 1, "test_accessed": False, "artifacts": artifacts},
    )
    return report_path


def _comparison(
    commit: str,
    evidence_path: Path,
    evidence: dict[str, Any],
    report_path: Path,
    checkpoints: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "task": "P1-A05-DEVELOPMENT-COMPARISON",
        "code": {"commit": commit, "working_tree_dirty": False},
        "inputs": {
            "candidate_evidence": {
                "path": evidence_path.name,
                "sha256": _file_sha256(evidence_path),
                "records_sha256": evidence["records_sha256"],
            },
            "candidate_standard_report": _record(report_path),
            "formal_training": {
                "summary": {"path": "summary.json", "bytes": 1, "sha256": "a" * 64},
                "run_manifest": {
                    "path": "manifest.json",
                    "bytes": 1,
                    "sha256": "b" * 64,
                },
                "implementation_review_sha256": "c" * 64,
                "checkpoints": checkpoints,
            },
            "test_accessed": False,
            "public_test": "forbidden",
        },
        "paired_development": {
            slice_id: {"candidate_mean_ratio": 0.72} for slice_id in SLICES
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


def _fixture(tmp_path: Path) -> dict[str, Any]:
    repository = tmp_path / "repository"
    commit = _git_repository(repository)
    contract = {
        "format_version": 1,
        "primary_policy": "masked_mlp",
        "policies": [
            {"id": "heft", "required_seeds": [0]},
            {"id": "masked_mlp", "required_seeds": SEEDS},
        ],
        "slices": [{"id": slice_id, "scenario_count": 1} for slice_id in SLICES],
        "modes": {"development": SLICES},
    }
    contract_path = _write_json(repository / "configs" / "contract.json", contract)
    contract_hash = _canonical_sha256(contract)

    checkpoints: dict[str, Any] = {}
    for seed in SEEDS:
        checkpoint = repository / "outputs" / "training" / f"seed_{seed}.npz"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint-{seed}\n".encode("ascii"))
        checkpoints[str(seed)] = {
            "path": checkpoint.relative_to(repository).as_posix(),
            "bytes": checkpoint.stat().st_size,
            "sha256": _file_sha256(checkpoint),
            "parameter_sha256": hashlib.sha256(
                f"parameter-{seed}".encode()
            ).hexdigest(),
        }

    roots = {
        "formal": repository / "outputs" / "formal",
        "replay": repository / "outputs" / "replay",
    }
    artifacts: dict[str, Any] = {}
    for label, runtime in (("formal", 1.0), ("replay", 9.0)):
        root = roots[label]
        evidence = _evidence(commit, contract_hash, runtime)
        evidence_path = _write_json(root / "development-evidence.json", evidence)
        report_dir = root / "report"
        report_path = _write_report(report_dir, evidence_path, evidence, runtime)
        comparison_path = _write_json(
            root / "comparison.json",
            _comparison(commit, evidence_path, evidence, report_path, checkpoints),
        )
        artifacts[label] = {
            "root": root,
            "evidence": evidence_path,
            "report_dir": report_dir,
            "report": report_path,
            "comparison": comparison_path,
        }
    output = repository / "outputs" / "review" / "receipt.json"
    kwargs = {
        "repository": repository,
        "remote_ref": "origin/main",
        "contract_path": contract_path,
        "formal_evidence_path": artifacts["formal"]["evidence"],
        "formal_report_dir": artifacts["formal"]["report_dir"],
        "formal_comparison_path": artifacts["formal"]["comparison"],
        "replay_evidence_path": artifacts["replay"]["evidence"],
        "replay_report_dir": artifacts["replay"]["report_dir"],
        "replay_comparison_path": artifacts["replay"]["comparison"],
        "output_path": output,
    }
    return {
        "repository": repository,
        "commit": commit,
        "artifacts": artifacts,
        "output": output,
        "kwargs": kwargs,
    }


def _rebind_replay_comparison(fixture: dict[str, Any]) -> None:
    replay = fixture["artifacts"]["replay"]
    evidence = json.loads(replay["evidence"].read_text(encoding="utf-8"))
    comparison = json.loads(replay["comparison"].read_text(encoding="utf-8"))
    comparison["inputs"]["candidate_evidence"].update(
        {
            "sha256": _file_sha256(replay["evidence"]),
            "records_sha256": evidence["records_sha256"],
        }
    )
    comparison["inputs"]["candidate_standard_report"] = _record(replay["report"])
    _write_json(replay["comparison"], comparison)


def test_independent_review_binds_normalized_replay(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = build_independent_review(**fixture["kwargs"])

    receipt = json.loads(result.read_text(encoding="utf-8"))
    assert receipt["decision"] == "approve_before_g3"
    assert receipt["reviewer"] == "B"
    assert receipt["candidate_commit"] == fixture["commit"]
    assert receipt["normalized_equivalence"]["record_count"] == 24
    assert receipt["assertions"]["normalized_scheduling_records_equal"] is True
    assert receipt["assertions"]["normalized_csvs_equal"] is True
    assert len(receipt["checkpoints"]) == 5


def test_independent_review_refuses_non_runtime_record_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    replay = fixture["artifacts"]["replay"]
    evidence = json.loads(replay["evidence"].read_text(encoding="utf-8"))
    evidence["records"][1]["score_ratio"] += 0.1
    evidence["records_sha256"] = _canonical_sha256(evidence["records"])
    _write_json(replay["evidence"], evidence)
    _rebind_replay_comparison(fixture)

    with pytest.raises(P1A05ReviewError, match="scheduling records differ"):
        build_independent_review(**fixture["kwargs"])


def test_independent_review_refuses_non_runtime_csv_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    replay = fixture["artifacts"]["replay"]
    csv_path = replay["report_dir"] / "evaluation_per_seed.csv"
    csv_path.write_text(
        "slice_id,policy,mean_ratio,runtime_mean_ms\n"
        "id_validation,masked_mlp,0.99,9.0\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest_path = replay["report_dir"] / "evaluation_report_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][csv_path.name] = {
        "bytes": csv_path.stat().st_size,
        "sha256": _file_sha256(csv_path),
    }
    _write_json(manifest_path, manifest)

    with pytest.raises(P1A05ReviewError, match="CSVs differ"):
        build_independent_review(**fixture["kwargs"])


def test_independent_review_requires_remote_immutable_commit(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    subprocess.run(
        ["git", "update-ref", "-d", "refs/remotes/origin/main"],
        cwd=fixture["repository"],
        check=True,
    )

    with pytest.raises(P1A05ReviewError, match="cannot inspect immutable remote ref"):
        build_independent_review(**fixture["kwargs"])
