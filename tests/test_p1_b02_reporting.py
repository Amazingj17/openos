from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from trisched import cli
from trisched.reporting import (
    EvaluationReportError,
    build_evaluation_report,
    canonical_json_sha256,
    claim_public_test_gate,
    file_sha256,
    load_evaluation_contract,
    scenario_set_sha256,
)


def tiny_contract() -> dict[str, object]:
    return {
        "format_version": 1,
        "contract_id": "p1_b02_test_v1",
        "primary_policy": "masked_mlp",
        "reference_policy": "heft",
        "tie_tolerance": 1e-9,
        "failure_penalty_ratio": 10.0,
        "metrics": {
            "score_ratio": "policy_makespan_over_heft_makespan",
            "runtime_ms": "wall_clock_scheduler_call_only",
            "runtime_quantiles": [0.5, 0.95],
            "timeout_error_code": "scheduler_timeout",
            "failures_remain_in_denominator": True,
        },
        "bootstrap": {
            "samples": 500,
            "seed": 17,
            "confidence_level": 0.95,
            "hierarchy": ["seed", "scenario"],
        },
        "policies": [
            {"id": "heft", "role": "baseline", "required_seeds": [0]},
            {
                "id": "masked_mlp",
                "role": "primary",
                "required_seeds": [11, 12],
            },
            {
                "id": "task_gnn",
                "role": "ablation",
                "required_seeds": [11],
            },
        ],
        "slices": [
            {
                "id": "id_validation",
                "role": "development",
                "scenario_count": 2,
                "definition": "two synthetic ID fixtures",
            },
            {
                "id": "public_test",
                "role": "final_test",
                "scenario_count": 2,
                "definition": "two synthetic test-gate fixtures",
            },
        ],
        "modes": {
            "development": ["id_validation"],
            "final_test": ["id_validation", "public_test"],
        },
        "test_gate": {
            "slice_id": "public_test",
            "policy": "one_time_final_only",
            "required_signers": ["A", "B"],
            "require_clean_commit": True,
            "receipt_name": "public_test_gate_receipt.json",
        },
    }


def write_json(path: Path, value: object) -> Path:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _scenario_rows(slice_id: str) -> list[dict[str, str]]:
    return [
        {
            "scenario_id": f"{slice_id}-case-{index}",
            "scenario_hash": hashlib.sha256(
                f"{slice_id}-case-{index}".encode("utf-8")
            ).hexdigest(),
        }
        for index in range(2)
    ]


def evidence_package(
    contract: dict[str, object], mode: str = "development"
) -> dict[str, object]:
    policies = {item["id"]: item for item in contract["policies"]}  # type: ignore[index]
    slices = contract["modes"][mode]  # type: ignore[index]
    manifests: dict[str, object] = {}
    records: list[dict[str, object]] = []
    primary_values = {
        (11, 0): 0.8,
        (11, 1): 1.2,
        (12, 0): 0.7,
        (12, 1): 1.1,
    }
    for slice_id in slices:
        scenarios = _scenario_rows(slice_id)
        manifests[slice_id] = {
            "scenario_count": len(scenarios),
            "scenario_set_sha256": scenario_set_sha256(scenarios),
        }
        for policy_id, policy in policies.items():
            for seed in policy["required_seeds"]:
                for index, scenario in enumerate(scenarios):
                    if policy_id == "masked_mlp":
                        ratio = primary_values[(seed, index)]
                    elif policy_id == "heft":
                        ratio = 1.0
                    else:
                        ratio = 0.9 + 0.1 * index
                    records.append(
                        {
                            "slice_id": slice_id,
                            "policy": policy_id,
                            "seed": seed,
                            **scenario,
                            "status": "success",
                            "ratio": ratio,
                            "score_ratio": ratio,
                            "penalty_applied": False,
                            "runtime_ms": 1.0 + index,
                            "illegal_action_count": 0,
                            "error_code": None,
                        }
                    )
    return {
        "format_version": 1,
        "mode": mode,
        "contract_sha256": canonical_json_sha256(contract),
        "code": {"commit": "a" * 40, "working_tree_dirty": False},
        "test_accessed": mode == "final_test",
        "slice_manifests": manifests,
        "records": records,
        "records_sha256": canonical_json_sha256(records),
    }


def test_tracked_p1_b02_contract_is_strict_and_five_seed() -> None:
    contract = load_evaluation_contract(Path("configs/p1_b02_evaluation_contract.json"))
    policies = {item["id"]: item for item in contract["policies"]}
    assert policies["masked_mlp"]["required_seeds"] == [
        20260717,
        20260718,
        20260719,
        20260720,
        20260721,
    ]
    assert contract["modes"]["development"] == [
        "id_validation",
        "ood_size",
        "ood_ccr",
        "ood_system",
    ]
    assert contract["test_gate"]["required_signers"] == ["A", "B"]
    assert contract["reference_policy"] == "heft"
    assert contract["metrics"] == {
        "score_ratio": "policy_makespan_over_heft_makespan",
        "runtime_ms": "wall_clock_scheduler_call_only",
        "runtime_quantiles": [0.5, 0.95],
        "timeout_error_code": "scheduler_timeout",
        "failures_remain_in_denominator": True,
    }


def test_development_report_is_deterministic_and_hashes_artifacts(
    tmp_path: Path,
) -> None:
    contract = tiny_contract()
    contract_path = write_json(tmp_path / "contract.json", contract)
    evidence = evidence_package(contract)
    evidence_path = write_json(tmp_path / "evidence.json", evidence)

    first_path = build_evaluation_report(
        contract_path, evidence_path, tmp_path / "first"
    )
    second_path = build_evaluation_report(
        contract_path, evidence_path, tmp_path / "second"
    )
    first = json.loads(first_path.read_text(encoding="utf-8"))
    second = json.loads(second_path.read_text(encoding="utf-8"))
    assert first == second
    assert first["report_scope"] == "development"
    assert first["evidence"]["test_accessed"] is False
    assert first["gate"]["release_publishable"] is False
    primary = first["slices"][0]["policies"]["masked_mlp"]
    assert primary["mean_ratio"] == pytest.approx(0.95)
    assert primary["success_count"] == 4
    assert primary["failure_count"] == 0
    assert primary["valid_schedule_rate"] == pytest.approx(1.0)
    assert primary["illegal_record_rate"] == pytest.approx(0.0)
    assert primary["win_tie_loss_vs_reference"] == {
        "win": 2,
        "tie": 0,
        "loss": 2,
    }
    assert primary["runtime_ms"]["p50"] == pytest.approx(1.5)
    assert primary["hierarchical_bootstrap"]["samples"] == 500
    comparison = first["slices"][0]["primary_comparisons"]["heft"]
    assert comparison["mean_paired_delta"] == pytest.approx(-0.05)
    assert comparison["scenario_mean_win_tie_loss"] == {
        "win": 1,
        "tie": 0,
        "loss": 1,
    }
    assert comparison["all_seed_scenario_pairs"] is None
    assert comparison["hierarchical_paired_bootstrap"]["samples"] == 500

    csv_rows = list(
        csv.DictReader(
            (tmp_path / "first" / "evaluation_per_slice.csv").open(
                encoding="utf-8", newline=""
            )
        )
    )
    assert len(csv_rows) == 3
    manifest = json.loads(
        (tmp_path / "first" / "evaluation_report_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["test_accessed"] is False
    assert set(manifest["artifacts"]) == {
        "evaluation_report.json",
        "evaluation_per_slice.csv",
        "evaluation_primary_comparisons.csv",
    }
    for name, metadata in manifest["artifacts"].items():
        artifact = tmp_path / "first" / name
        assert artifact.stat().st_size == metadata["bytes"]
        assert file_sha256(artifact) == metadata["sha256"]


def mutate_and_write(
    tmp_path: Path,
    contract: dict[str, object],
    evidence: dict[str, object],
    *,
    update_records_hash: bool,
) -> tuple[Path, Path]:
    if update_records_hash:
        evidence["records_sha256"] = canonical_json_sha256(evidence["records"])
    return (
        write_json(tmp_path / "contract.json", contract),
        write_json(tmp_path / "evidence.json", evidence),
    )


def test_report_rejects_records_hash_drift(tmp_path: Path) -> None:
    contract = tiny_contract()
    evidence = evidence_package(contract)
    evidence["records"][0]["score_ratio"] = 0.5  # type: ignore[index]
    contract_path, evidence_path = mutate_and_write(
        tmp_path, contract, evidence, update_records_hash=False
    )
    with pytest.raises(EvaluationReportError) as captured:
        build_evaluation_report(contract_path, evidence_path, tmp_path / "out")
    assert captured.value.code == "report_hash"
    assert captured.value.path == "$.records_sha256"


def test_report_rejects_missing_seed_scenario_pair(tmp_path: Path) -> None:
    contract = tiny_contract()
    evidence = evidence_package(contract)
    evidence["records"] = [
        row
        for row in evidence["records"]  # type: ignore[union-attr]
        if not (
            row["policy"] == "masked_mlp"
            and row["seed"] == 12
            and row["scenario_id"] == "id_validation-case-1"
        )
    ]
    contract_path, evidence_path = mutate_and_write(
        tmp_path, contract, evidence, update_records_hash=True
    )
    with pytest.raises(EvaluationReportError) as captured:
        build_evaluation_report(contract_path, evidence_path, tmp_path / "out")
    assert captured.value.code == "report_seed_set"
    assert captured.value.details["seed"] == 12


def test_report_rejects_scenario_hash_mismatch(tmp_path: Path) -> None:
    contract = tiny_contract()
    evidence = evidence_package(contract)
    evidence["records"][1]["scenario_hash"] = "b" * 64  # type: ignore[index]
    contract_path, evidence_path = mutate_and_write(
        tmp_path, contract, evidence, update_records_hash=True
    )
    with pytest.raises(EvaluationReportError) as captured:
        build_evaluation_report(contract_path, evidence_path, tmp_path / "out")
    assert captured.value.code == "report_scenario_set"


def test_report_rejects_incorrect_failure_penalty(tmp_path: Path) -> None:
    contract = tiny_contract()
    evidence = evidence_package(contract)
    row = evidence["records"][2]  # type: ignore[index]
    row.update(
        {
            "status": "failure",
            "ratio": None,
            "score_ratio": 9.0,
            "penalty_applied": True,
            "error_code": "scheduler_timeout",
        }
    )
    contract_path, evidence_path = mutate_and_write(
        tmp_path, contract, evidence, update_records_hash=True
    )
    with pytest.raises(EvaluationReportError) as captured:
        build_evaluation_report(contract_path, evidence_path, tmp_path / "out")
    assert captured.value.code == "report_failure_accounting"


def test_report_keeps_valid_failure_in_all_denominators(tmp_path: Path) -> None:
    contract = tiny_contract()
    evidence = evidence_package(contract)
    failed_row = next(
        row for row in evidence["records"] if row["policy"] == "task_gnn"  # type: ignore[union-attr]
    )
    failed_row.update(
        {
            "status": "failure",
            "ratio": None,
            "score_ratio": 10.0,
            "penalty_applied": True,
            "error_code": "scheduler_timeout",
            "illegal_action_count": 1,
        }
    )
    contract_path, evidence_path = mutate_and_write(
        tmp_path, contract, evidence, update_records_hash=True
    )
    report_path = build_evaluation_report(
        contract_path,
        evidence_path,
        tmp_path / "out",
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = report["slices"][0]["policies"]["task_gnn"]
    assert metrics["record_count"] == 2
    assert metrics["success_count"] == 1
    assert metrics["failure_count"] == 1
    assert metrics["failure_rate"] == pytest.approx(0.5)
    assert metrics["valid_schedule_rate"] == pytest.approx(0.5)
    assert metrics["illegal_action_count"] == 1
    assert metrics["illegal_record_count"] == 1
    assert metrics["illegal_record_rate"] == pytest.approx(0.5)
    assert metrics["timeout_count"] == 1
    assert metrics["timeout_rate"] == pytest.approx(0.5)
    assert metrics["mean_ratio"] == pytest.approx(5.5)


@pytest.mark.parametrize("reference_status", ["failure", "wrong_ratio"])
def test_report_rejects_invalid_reference_rows(
    tmp_path: Path,
    reference_status: str,
) -> None:
    contract = tiny_contract()
    evidence = evidence_package(contract)
    reference_row = next(
        row for row in evidence["records"] if row["policy"] == "heft"  # type: ignore[union-attr]
    )
    if reference_status == "failure":
        reference_row.update(
            {
                "status": "failure",
                "ratio": None,
                "score_ratio": 10.0,
                "penalty_applied": True,
                "error_code": "scheduler_timeout",
            }
        )
    else:
        reference_row.update({"ratio": 0.99, "score_ratio": 0.99})
    contract_path, evidence_path = mutate_and_write(
        tmp_path, contract, evidence, update_records_hash=True
    )
    with pytest.raises(EvaluationReportError) as captured:
        build_evaluation_report(contract_path, evidence_path, tmp_path / "out")
    assert captured.value.code == "report_reference_policy"


def test_report_emits_exact_pairs_for_identical_seed_sets(tmp_path: Path) -> None:
    contract = tiny_contract()
    task_gnn = next(
        item for item in contract["policies"] if item["id"] == "task_gnn"  # type: ignore[union-attr]
    )
    task_gnn["required_seeds"] = [11, 12]
    contract_path = write_json(tmp_path / "contract.json", contract)
    evidence_path = write_json(
        tmp_path / "evidence.json",
        evidence_package(contract),
    )
    report_path = build_evaluation_report(
        contract_path,
        evidence_path,
        tmp_path / "out",
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    exact = report["slices"][0]["primary_comparisons"]["task_gnn"][
        "all_seed_scenario_pairs"
    ]
    assert exact["pair_count"] == 4
    assert exact["mean_delta"] == pytest.approx(0.0)
    assert exact["win_tie_loss"] == {"win": 2, "tie": 0, "loss": 2}


def test_report_rejects_output_path_that_is_a_file(tmp_path: Path) -> None:
    contract = tiny_contract()
    contract_path = write_json(tmp_path / "contract.json", contract)
    evidence_path = write_json(
        tmp_path / "evidence.json",
        evidence_package(contract),
    )
    output_path = write_json(tmp_path / "occupied.json", {})
    with pytest.raises(EvaluationReportError) as captured:
        build_evaluation_report(contract_path, evidence_path, output_path)
    assert captured.value.code == "report_output"


def test_development_mode_rejects_test_access_claim(tmp_path: Path) -> None:
    contract = tiny_contract()
    evidence = evidence_package(contract)
    evidence["test_accessed"] = True
    contract_path, evidence_path = mutate_and_write(
        tmp_path, contract, evidence, update_records_hash=False
    )
    with pytest.raises(EvaluationReportError) as captured:
        build_evaluation_report(contract_path, evidence_path, tmp_path / "out")
    assert captured.value.code == "report_test_gate"


def authorization(contract: dict[str, object]) -> dict[str, object]:
    return {
        "format_version": 1,
        "purpose": "public_test_once",
        "contract_sha256": canonical_json_sha256(contract),
        "release_commit": "a" * 40,
        "working_tree_dirty": False,
        "test_slice_id": "public_test",
        "authorized_by": ["A", "B"],
        "authorized_at_utc": "2026-07-17T00:00:00Z",
        "authorization_nonce": "c" * 64,
    }


def test_public_test_gate_is_one_time_and_binds_final_report(
    tmp_path: Path,
) -> None:
    contract = tiny_contract()
    contract_path = write_json(tmp_path / "contract.json", contract)
    authorization_path = write_json(
        tmp_path / "authorization.json", authorization(contract)
    )
    receipt_path = tmp_path / "public_test_gate_receipt.json"
    assert (
        claim_public_test_gate(contract_path, authorization_path, receipt_path)
        == receipt_path
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "claimed"
    assert receipt["test_accessed"] is False
    with pytest.raises(EvaluationReportError) as captured:
        claim_public_test_gate(contract_path, authorization_path, receipt_path)
    assert captured.value.code == "report_test_gate_consumed"

    final_evidence = evidence_package(contract, mode="final_test")
    evidence_path = write_json(tmp_path / "final_evidence.json", final_evidence)
    report_path = build_evaluation_report(
        contract_path,
        evidence_path,
        tmp_path / "final_report",
        test_receipt_path=receipt_path,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["report_scope"] == "final_test"
    assert report["evidence"]["test_accessed"] is True
    assert report["gate"]["release_publishable"] is True
    assert report["test_gate_receipt"]["sha256"] == file_sha256(receipt_path)


def test_public_test_gate_reports_invalid_commit_as_gate_error(tmp_path: Path) -> None:
    contract = tiny_contract()
    contract_path = write_json(tmp_path / "contract.json", contract)
    authorization_payload = authorization(contract)
    authorization_payload["release_commit"] = "not-a-commit"
    authorization_path = write_json(
        tmp_path / "authorization.json",
        authorization_payload,
    )
    with pytest.raises(EvaluationReportError) as captured:
        claim_public_test_gate(
            contract_path,
            authorization_path,
            tmp_path / "receipt.json",
        )
    assert captured.value.code == "report_test_gate"
    assert captured.value.path == "$.release_commit"


def test_final_report_without_receipt_is_rejected(tmp_path: Path) -> None:
    contract = tiny_contract()
    contract_path = write_json(tmp_path / "contract.json", contract)
    evidence_path = write_json(
        tmp_path / "evidence.json",
        evidence_package(contract, mode="final_test"),
    )
    with pytest.raises(EvaluationReportError) as captured:
        build_evaluation_report(contract_path, evidence_path, tmp_path / "out")
    assert captured.value.code == "report_test_gate"


def test_cli_returns_structured_report_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    contract = tiny_contract()
    evidence = evidence_package(contract)
    evidence["records_sha256"] = "0" * 64
    contract_path = write_json(tmp_path / "contract.json", contract)
    evidence_path = write_json(tmp_path / "evidence.json", evidence)
    result = cli.main(
        [
            "build-report",
            "--contract",
            str(contract_path),
            "--evidence",
            str(evidence_path),
            "--output",
            str(tmp_path / "out"),
        ]
    )
    captured = capsys.readouterr()
    assert result == 4
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"]["code"] == "report_hash"
    assert payload["error"]["path"] == "$.records_sha256"
