from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


class EvaluationReportError(RuntimeError):
    """Stable diagnostic for P1-B02 contract and report failures."""

    def __init__(
        self,
        code: str,
        path: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.path = path
        self.message = message
        self.details = {} if details is None else details

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "path": self.path,
            "message": self.message,
            "details": self.details,
        }


def _fail(
    code: str,
    path: str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
) -> None:
    raise EvaluationReportError(code, path, message, details=details)


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scenario_set_sha256(rows: list[dict[str, str]]) -> str:
    normalized = sorted(
        (
            {"scenario_id": row["scenario_id"], "scenario_hash": row["scenario_hash"]}
            for row in rows
        ),
        key=lambda row: row["scenario_id"],
    )
    return canonical_json_sha256(normalized)


def _load_object(path: str | Path, *, code: str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail(code, "$", f"could not read JSON object: {error}")
    if not isinstance(value, dict):
        _fail(code, "$", "expected a JSON object")
    return value


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _number(value: Any, path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("report_contract", path, "expected a finite number")
    result = float(value)
    if not math.isfinite(result):
        _fail("report_contract", path, "expected a finite number")
    if minimum is not None and result < minimum:
        _fail("report_contract", path, f"expected a value >= {minimum}")
    return result


def _identifier(value: Any, path: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
            for character in value
        )
    ):
        _fail(
            "report_contract",
            path,
            "expected a non-empty lowercase identifier",
        )
    return value


def _hex(value: Any, path: str, length: int, *, code: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(code, path, f"expected {length} lowercase hex characters")
    return value


def _json_filename(value: Any, path: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith(".")
        or not value.endswith(".json")
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789._-"
            for character in value
        )
        or "/" in value
        or "\\" in value
    ):
        _fail(
            "report_contract",
            path,
            "expected a safe lowercase JSON basename",
        )
    return value


def _utc_timestamp(value: Any, path: str, *, code: str) -> str:
    if not isinstance(value, str):
        _fail(code, path, "expected UTC timestamp YYYY-MM-DDTHH:MM:SSZ")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        _fail(code, path, "expected UTC timestamp YYYY-MM-DDTHH:MM:SSZ")
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        _fail(code, path, "expected canonical UTC timestamp YYYY-MM-DDTHH:MM:SSZ")
    return value


def load_evaluation_contract(path: str | Path) -> dict[str, Any]:
    contract = _load_object(path, code="report_contract")
    if contract.get("format_version") != 1:
        _fail("report_contract", "$.format_version", "expected format version 1")
    _identifier(contract.get("contract_id"), "$.contract_id")
    primary_policy = _identifier(contract.get("primary_policy"), "$.primary_policy")
    reference_policy = _identifier(
        contract.get("reference_policy"),
        "$.reference_policy",
    )
    tie_tolerance = _number(contract.get("tie_tolerance"), "$.tie_tolerance")
    if tie_tolerance <= 0.0:
        _fail("report_contract", "$.tie_tolerance", "expected a positive value")
    failure_penalty = _number(
        contract.get("failure_penalty_ratio"),
        "$.failure_penalty_ratio",
    )
    if failure_penalty <= 1.0:
        _fail(
            "report_contract",
            "$.failure_penalty_ratio",
            "expected a value > 1",
        )

    metrics = contract.get("metrics")
    expected_metrics = {
        "score_ratio": "policy_makespan_over_heft_makespan",
        "runtime_ms": "wall_clock_scheduler_call_only",
        "runtime_quantiles": [0.5, 0.95],
        "timeout_error_code": "scheduler_timeout",
        "failures_remain_in_denominator": True,
    }
    if metrics != expected_metrics:
        _fail(
            "report_contract",
            "$.metrics",
            "metrics must match the frozen ratio/runtime/failure definition",
            details={"expected": expected_metrics},
        )

    bootstrap = contract.get("bootstrap")
    if not isinstance(bootstrap, dict):
        _fail("report_contract", "$.bootstrap", "expected an object")
    samples = bootstrap.get("samples")
    seed = bootstrap.get("seed")
    if not _is_int(samples) or samples <= 0:
        _fail("report_contract", "$.bootstrap.samples", "expected a positive integer")
    if not _is_int(seed):
        _fail("report_contract", "$.bootstrap.seed", "expected an integer")
    confidence = _number(
        bootstrap.get("confidence_level"),
        "$.bootstrap.confidence_level",
    )
    if not 0.0 < confidence < 1.0:
        _fail(
            "report_contract",
            "$.bootstrap.confidence_level",
            "expected a value between zero and one",
        )
    if bootstrap.get("hierarchy") != ["seed", "scenario"]:
        _fail(
            "report_contract",
            "$.bootstrap.hierarchy",
            "expected ['seed', 'scenario']",
        )

    policies = contract.get("policies")
    if not isinstance(policies, list) or not policies:
        _fail("report_contract", "$.policies", "expected a non-empty array")
    policy_ids: list[str] = []
    primary_count = 0
    for index, policy in enumerate(policies):
        path_prefix = f"$.policies[{index}]"
        if not isinstance(policy, dict):
            _fail("report_contract", path_prefix, "expected an object")
        policy_id = _identifier(policy.get("id"), f"{path_prefix}.id")
        if policy_id in policy_ids:
            _fail("report_contract", f"{path_prefix}.id", "duplicate policy id")
        policy_ids.append(policy_id)
        role = policy.get("role")
        if role not in {"primary", "baseline", "ablation"}:
            _fail("report_contract", f"{path_prefix}.role", "invalid policy role")
        if role == "primary":
            primary_count += 1
        seeds = policy.get("required_seeds")
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(not _is_int(value) for value in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            _fail(
                "report_contract",
                f"{path_prefix}.required_seeds",
                "expected unique integer seeds",
            )
    if primary_count != 1 or primary_policy not in policy_ids:
        _fail(
            "report_contract",
            "$.primary_policy",
            "primary policy must name the single policy with role=primary",
        )
    if (
        next(item for item in policies if item["id"] == primary_policy).get("role")
        != "primary"
    ):
        _fail(
            "report_contract",
            "$.primary_policy",
            "primary policy has the wrong role",
        )
    if (
        reference_policy not in policy_ids
        or next(item for item in policies if item["id"] == reference_policy).get("role")
        != "baseline"
    ):
        _fail(
            "report_contract",
            "$.reference_policy",
            "reference policy must name a policy with role=baseline",
        )

    slices = contract.get("slices")
    if not isinstance(slices, list) or not slices:
        _fail("report_contract", "$.slices", "expected a non-empty array")
    slice_ids: list[str] = []
    final_slices: list[str] = []
    for index, item in enumerate(slices):
        path_prefix = f"$.slices[{index}]"
        if not isinstance(item, dict):
            _fail("report_contract", path_prefix, "expected an object")
        slice_id = _identifier(item.get("id"), f"{path_prefix}.id")
        if slice_id in slice_ids:
            _fail("report_contract", f"{path_prefix}.id", "duplicate slice id")
        slice_ids.append(slice_id)
        role = item.get("role")
        if role not in {"development", "development_ood", "final_test"}:
            _fail("report_contract", f"{path_prefix}.role", "invalid slice role")
        if role == "final_test":
            final_slices.append(slice_id)
        count = item.get("scenario_count")
        if not _is_int(count) or count < 2:
            _fail(
                "report_contract",
                f"{path_prefix}.scenario_count",
                "expected an integer >= 2",
            )
        if not isinstance(item.get("definition"), str) or not item["definition"]:
            _fail(
                "report_contract",
                f"{path_prefix}.definition",
                "expected a non-empty definition",
            )

    modes = contract.get("modes")
    if not isinstance(modes, dict) or set(modes) != {"development", "final_test"}:
        _fail(
            "report_contract",
            "$.modes",
            "expected development and final_test modes",
        )
    for mode, required in modes.items():
        if (
            not isinstance(required, list)
            or not required
            or len(set(required)) != len(required)
            or any(value not in slice_ids for value in required)
        ):
            _fail(
                "report_contract",
                f"$.modes.{mode}",
                "expected unique known slice ids",
            )
    if any(value in final_slices for value in modes["development"]):
        _fail(
            "report_contract",
            "$.modes.development",
            "development mode cannot contain a final-test slice",
        )
    if set(final_slices) - set(modes["final_test"]):
        _fail(
            "report_contract",
            "$.modes.final_test",
            "final mode must contain every final-test slice",
        )

    gate = contract.get("test_gate")
    if not isinstance(gate, dict):
        _fail("report_contract", "$.test_gate", "expected an object")
    gate_slice = _identifier(gate.get("slice_id"), "$.test_gate.slice_id")
    if final_slices != [gate_slice]:
        _fail(
            "report_contract",
            "$.test_gate.slice_id",
            "gate must name the single final-test slice",
        )
    if gate.get("policy") != "one_time_final_only":
        _fail(
            "report_contract",
            "$.test_gate.policy",
            "expected one_time_final_only",
        )
    signers = gate.get("required_signers")
    if (
        not isinstance(signers, list)
        or not signers
        or any(not isinstance(value, str) or not value for value in signers)
        or len(set(signers)) != len(signers)
    ):
        _fail(
            "report_contract",
            "$.test_gate.required_signers",
            "expected unique signer names",
        )
    if gate.get("require_clean_commit") is not True:
        _fail(
            "report_contract",
            "$.test_gate.require_clean_commit",
            "final test must require a clean commit",
        )
    _json_filename(gate.get("receipt_name"), "$.test_gate.receipt_name")
    if gate.get("authorization_time_format") != "utc_seconds_z":
        _fail(
            "report_contract",
            "$.test_gate.authorization_time_format",
            "expected utc_seconds_z",
        )
    return contract


def claim_public_test_gate(
    contract_path: str | Path,
    authorization_path: str | Path,
    receipt_path: str | Path,
) -> Path:
    contract_source = Path(contract_path)
    authorization_source = Path(authorization_path)
    receipt = Path(receipt_path)
    contract = load_evaluation_contract(contract_source)
    contract_hash = canonical_json_sha256(contract)
    expected_receipt_name = contract["test_gate"]["receipt_name"]
    if receipt.name != expected_receipt_name:
        _fail(
            "report_test_gate",
            "$receipt",
            "receipt basename does not match the frozen contract",
            details={
                "expected": expected_receipt_name,
                "actual": receipt.name,
            },
        )
    authorization = _load_object(authorization_source, code="report_test_gate")
    if authorization.get("format_version") != 1:
        _fail("report_test_gate", "$.format_version", "expected format version 1")
    if authorization.get("purpose") != "public_test_once":
        _fail("report_test_gate", "$.purpose", "expected public_test_once")
    if authorization.get("contract_sha256") != contract_hash:
        _fail("report_test_gate", "$.contract_sha256", "contract hash mismatch")
    commit = _hex(
        authorization.get("release_commit"),
        "$.release_commit",
        40,
        code="report_test_gate",
    )
    if authorization.get("working_tree_dirty") is not False:
        _fail(
            "report_test_gate",
            "$.working_tree_dirty",
            "public test requires a clean release commit",
        )
    if authorization.get("test_slice_id") != contract["test_gate"]["slice_id"]:
        _fail("report_test_gate", "$.test_slice_id", "test slice mismatch")
    signers = authorization.get("authorized_by")
    if signers != contract["test_gate"]["required_signers"]:
        _fail(
            "report_test_gate",
            "$.authorized_by",
            "authorization signers do not match the frozen order",
        )
    nonce = _hex(
        authorization.get("authorization_nonce"),
        "$.authorization_nonce",
        64,
        code="report_test_gate",
    )
    authorized_at_utc = _utc_timestamp(
        authorization.get("authorized_at_utc"),
        "$.authorized_at_utc",
        code="report_test_gate",
    )
    if receipt.exists():
        _fail(
            "report_test_gate_consumed",
            "$receipt",
            "public-test gate receipt already exists",
            details={"receipt": receipt.name},
        )
    payload = {
        "format_version": 1,
        "mode": "public_test_gate_receipt",
        "status": "claimed",
        "contract_id": contract["contract_id"],
        "contract_sha256": contract_hash,
        "release_commit": commit,
        "test_slice_id": contract["test_gate"]["slice_id"],
        "receipt_name": expected_receipt_name,
        "authorized_by": signers,
        "authorized_at_utc": authorized_at_utc,
        "authorization_nonce": nonce,
        "authorization_file_sha256": file_sha256(authorization_source),
        "test_accessed": False,
        "claim_boundary": (
            "claim immediately before the public-test loader; a second claim at "
            "this receipt path is refused"
        ),
    }
    try:
        receipt.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        _fail("report_test_gate_write", "$receipt", str(error))
    created = False
    try:
        with receipt.open("x", encoding="utf-8", newline="\n") as handle:
            created = True
            handle.write(
                json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        _fail(
            "report_test_gate_consumed",
            "$receipt",
            "public-test gate receipt already exists",
            details={"receipt": receipt.name},
        )
    except (OSError, ValueError, TypeError) as error:
        if created:
            try:
                receipt.unlink(missing_ok=True)
            except OSError:
                pass
        _fail("report_test_gate_write", "$receipt", str(error))
    return receipt


def _contract_maps(
    contract: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    return (
        {item["id"]: item for item in contract["policies"]},
        {item["id"]: item for item in contract["slices"]},
    )


def _validate_test_receipt(
    receipt_path: str | Path | None,
    *,
    contract: dict[str, Any],
    contract_hash: str,
    commit: str,
) -> dict[str, Any]:
    if receipt_path is None:
        _fail(
            "report_test_gate",
            "$receipt",
            "final-test evidence requires a gate receipt",
        )
    expected_receipt_name = contract["test_gate"]["receipt_name"]
    actual_receipt_name = Path(receipt_path).name
    if actual_receipt_name != expected_receipt_name:
        _fail(
            "report_test_gate",
            "$receipt",
            "receipt basename does not match the frozen contract",
            details={
                "expected": expected_receipt_name,
                "actual": actual_receipt_name,
            },
        )
    receipt = _load_object(receipt_path, code="report_test_gate")
    _utc_timestamp(
        receipt.get("authorized_at_utc"),
        "$receipt.authorized_at_utc",
        code="report_test_gate",
    )
    expected = {
        "format_version": 1,
        "mode": "public_test_gate_receipt",
        "status": "claimed",
        "contract_sha256": contract_hash,
        "release_commit": commit,
        "test_slice_id": contract["test_gate"]["slice_id"],
        "receipt_name": expected_receipt_name,
        "authorized_by": contract["test_gate"]["required_signers"],
        "test_accessed": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            _fail(
                "report_test_gate",
                f"$receipt.{key}",
                "gate receipt does not match final evidence",
            )
    return receipt


def _validate_evidence(
    evidence_path: str | Path,
    contract: dict[str, Any],
    contract_hash: str,
    receipt_path: str | Path | None,
) -> tuple[
    dict[str, Any],
    dict[tuple[str, str, int, str], dict[str, Any]],
    dict[str, list[dict[str, str]]],
    dict[str, dict[str, Any]] | None,
]:
    evidence = _load_object(evidence_path, code="report_evidence")
    if evidence.get("format_version") != 1:
        _fail("report_evidence", "$.format_version", "expected format version 1")
    mode = evidence.get("mode")
    if mode not in {"development", "final_test"}:
        _fail("report_evidence", "$.mode", "invalid report mode")
    if evidence.get("contract_sha256") != contract_hash:
        _fail("report_hash", "$.contract_sha256", "contract hash mismatch")
    code = evidence.get("code")
    if not isinstance(code, dict):
        _fail("report_evidence", "$.code", "expected an object")
    commit = _hex(
        code.get("commit"),
        "$.code.commit",
        40,
        code="report_evidence",
    )
    if not isinstance(code.get("working_tree_dirty"), bool):
        _fail(
            "report_evidence",
            "$.code.working_tree_dirty",
            "expected a boolean",
        )
    expected_test_accessed = mode == "final_test"
    if evidence.get("test_accessed") is not expected_test_accessed:
        _fail(
            "report_test_gate",
            "$.test_accessed",
            f"{mode} requires test_accessed={str(expected_test_accessed).lower()}",
        )
    if mode == "final_test" and code["working_tree_dirty"]:
        _fail(
            "report_test_gate",
            "$.code.working_tree_dirty",
            "final test requires a clean release commit",
        )

    records = evidence.get("records")
    if not isinstance(records, list) or not records:
        _fail("report_evidence", "$.records", "expected a non-empty array")
    try:
        records_hash = canonical_json_sha256(records)
    except (TypeError, ValueError) as error:
        _fail("report_evidence", "$.records", str(error))
    if evidence.get("records_sha256") != records_hash:
        _fail("report_hash", "$.records_sha256", "records hash mismatch")

    policies, slices = _contract_maps(contract)
    required_slices = contract["modes"][mode]
    observed_slices: set[str] = set()
    scenario_hashes: dict[str, dict[str, str]] = {
        slice_id: {} for slice_id in required_slices
    }
    indexed: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    penalty = float(contract["failure_penalty_ratio"])
    tolerance = float(contract["tie_tolerance"])
    for index, record in enumerate(records):
        path_prefix = f"$.records[{index}]"
        if not isinstance(record, dict):
            _fail("report_evidence", path_prefix, "expected an object")
        slice_id = record.get("slice_id")
        policy_id = record.get("policy")
        seed = record.get("seed")
        scenario_id = record.get("scenario_id")
        scenario_hash = record.get("scenario_hash")
        if slice_id not in required_slices:
            code_name = (
                "report_test_gate"
                if slice_id in slices and slices[slice_id]["role"] == "final_test"
                else "report_slice_set"
            )
            _fail(code_name, f"{path_prefix}.slice_id", "slice is not allowed in mode")
        if policy_id not in policies:
            _fail(
                "report_policy_set",
                f"{path_prefix}.policy",
                "unknown policy",
            )
        if not _is_int(seed):
            _fail("report_seed_set", f"{path_prefix}.seed", "expected an integer")
        if not isinstance(scenario_id, str) or not scenario_id:
            _fail(
                "report_scenario_set",
                f"{path_prefix}.scenario_id",
                "expected a non-empty scenario id",
            )
        _hex(
            scenario_hash,
            f"{path_prefix}.scenario_hash",
            64,
            code="report_scenario_set",
        )
        if seed not in policies[policy_id]["required_seeds"]:
            _fail(
                "report_seed_set",
                f"{path_prefix}.seed",
                "seed is not frozen for this policy",
            )
        observed_slices.add(slice_id)
        existing_hash = scenario_hashes[slice_id].get(scenario_id)
        if existing_hash is not None and existing_hash != scenario_hash:
            _fail(
                "report_scenario_set",
                f"{path_prefix}.scenario_hash",
                "scenario hash differs across paired records",
            )
        scenario_hashes[slice_id][scenario_id] = scenario_hash
        key = (slice_id, policy_id, seed, scenario_id)
        if key in indexed:
            _fail("report_evidence", path_prefix, "duplicate seed-scenario record")

        status = record.get("status")
        ratio = record.get("ratio")
        score_ratio = record.get("score_ratio")
        runtime_ms = record.get("runtime_ms")
        penalty_applied = record.get("penalty_applied")
        error_code = record.get("error_code")
        illegal_count = record.get("illegal_action_count")
        if (
            isinstance(score_ratio, bool)
            or not isinstance(score_ratio, (int, float))
            or not math.isfinite(float(score_ratio))
            or float(score_ratio) <= 0.0
        ):
            _fail(
                "report_failure_accounting",
                f"{path_prefix}.score_ratio",
                "score ratio must be finite and positive",
            )
        if (
            isinstance(runtime_ms, bool)
            or not isinstance(runtime_ms, (int, float))
            or not math.isfinite(float(runtime_ms))
            or float(runtime_ms) < 0.0
        ):
            _fail(
                "report_evidence",
                f"{path_prefix}.runtime_ms",
                "runtime must be finite and non-negative",
            )
        if not _is_int(illegal_count) or illegal_count < 0:
            _fail(
                "report_evidence",
                f"{path_prefix}.illegal_action_count",
                "illegal-action count must be a non-negative integer",
            )
        if status == "success":
            if (
                isinstance(ratio, bool)
                or not isinstance(ratio, (int, float))
                or not math.isfinite(float(ratio))
                or float(ratio) <= 0.0
                or not math.isclose(
                    float(ratio),
                    float(score_ratio),
                    rel_tol=0.0,
                    abs_tol=tolerance,
                )
                or penalty_applied is not False
                or error_code not in {None, ""}
                or illegal_count != 0
            ):
                _fail(
                    "report_failure_accounting",
                    path_prefix,
                    "success row does not match raw/score/error legality rules",
                )
        elif status == "failure":
            if (
                ratio is not None
                or not math.isclose(
                    float(score_ratio),
                    penalty,
                    rel_tol=0.0,
                    abs_tol=tolerance,
                )
                or penalty_applied is not True
                or not isinstance(error_code, str)
                or not error_code
            ):
                _fail(
                    "report_failure_accounting",
                    path_prefix,
                    "failure row does not match frozen penalty/error rules",
                )
        else:
            _fail(
                "report_failure_accounting",
                f"{path_prefix}.status",
                "status must be success or failure",
            )
        if policy_id == contract["reference_policy"] and (
            status != "success"
            or not math.isclose(
                float(ratio),
                1.0,
                rel_tol=0.0,
                abs_tol=tolerance,
            )
            or not math.isclose(
                float(score_ratio),
                1.0,
                rel_tol=0.0,
                abs_tol=tolerance,
            )
            or illegal_count != 0
        ):
            _fail(
                "report_reference_policy",
                path_prefix,
                "reference policy must succeed legally with ratio=score_ratio=1",
            )
        indexed[key] = record

    if observed_slices != set(required_slices):
        _fail(
            "report_slice_set",
            "$.records",
            "evidence does not contain the exact required slice set",
            details={
                "expected": required_slices,
                "actual": sorted(observed_slices),
            },
        )

    normalized_scenarios: dict[str, list[dict[str, str]]] = {}
    slice_manifests = evidence.get("slice_manifests")
    if not isinstance(slice_manifests, dict):
        _fail("report_evidence", "$.slice_manifests", "expected an object")
    if set(slice_manifests) != set(required_slices):
        _fail(
            "report_slice_set",
            "$.slice_manifests",
            "slice manifest keys differ from the frozen mode",
        )
    for slice_id in required_slices:
        items = [
            {"scenario_id": scenario_id, "scenario_hash": scenario_hash}
            for scenario_id, scenario_hash in scenario_hashes[slice_id].items()
        ]
        items.sort(key=lambda item: item["scenario_id"])
        expected_count = int(slices[slice_id]["scenario_count"])
        if len(items) != expected_count:
            _fail(
                "report_scenario_set",
                f"$.slice_manifests.{slice_id}.scenario_count",
                "scenario count differs from contract",
                details={"expected": expected_count, "actual": len(items)},
            )
        expected_hash = scenario_set_sha256(items)
        manifest = slice_manifests[slice_id]
        if (
            not isinstance(manifest, dict)
            or manifest.get("scenario_count") != expected_count
            or manifest.get("scenario_set_sha256") != expected_hash
        ):
            _fail(
                "report_hash",
                f"$.slice_manifests.{slice_id}",
                "slice manifest count/hash mismatch",
            )
        normalized_scenarios[slice_id] = items

    for slice_id in required_slices:
        scenario_ids = [item["scenario_id"] for item in normalized_scenarios[slice_id]]
        for policy_id, policy in policies.items():
            expected_seeds = policy["required_seeds"]
            for seed in expected_seeds:
                missing = [
                    scenario_id
                    for scenario_id in scenario_ids
                    if (slice_id, policy_id, seed, scenario_id) not in indexed
                ]
                if missing:
                    _fail(
                        "report_seed_set",
                        "$.records",
                        "missing frozen seed-scenario records",
                        details={
                            "slice_id": slice_id,
                            "policy": policy_id,
                            "seed": seed,
                            "missing": missing,
                        },
                    )
    expected_record_count = sum(
        len(policies[policy_id]["required_seeds"]) * len(normalized_scenarios[slice_id])
        for slice_id in required_slices
        for policy_id in policies
    )
    if len(indexed) != expected_record_count:
        _fail(
            "report_seed_set",
            "$.records",
            "record count differs from the frozen seed-scenario Cartesian product",
        )

    receipt = None
    if mode == "final_test":
        receipt = _validate_test_receipt(
            receipt_path,
            contract=contract,
            contract_hash=contract_hash,
            commit=commit,
        )
    elif receipt_path is not None:
        _fail(
            "report_test_gate",
            "$receipt",
            "development report must not use a public-test receipt",
        )
    return evidence, indexed, normalized_scenarios, receipt


def _aggregate_policy(
    records: list[dict[str, Any]],
    seeds: list[int],
    scenario_ids: list[str],
    *,
    bootstrap: dict[str, Any],
    tie_tolerance: float,
    timeout_error_code: str,
) -> dict[str, Any]:
    mapped = {(int(row["seed"]), row["scenario_id"]): row for row in records}
    ratios = np.asarray(
        [
            [
                float(mapped[(seed, scenario_id)]["score_ratio"])
                for scenario_id in scenario_ids
            ]
            for seed in seeds
        ],
        dtype=np.float64,
    )
    runtimes = np.asarray(
        [float(row["runtime_ms"]) for row in records], dtype=np.float64
    )
    flat = ratios.ravel()
    failure_count = sum(row["status"] == "failure" for row in records)
    success_count = len(records) - failure_count
    illegal_count = sum(int(row["illegal_action_count"]) for row in records)
    illegal_record_count = sum(int(row["illegal_action_count"]) > 0 for row in records)
    timeout_count = sum(row.get("error_code") == timeout_error_code for row in records)
    seed_means = np.mean(ratios, axis=1)
    rng = np.random.default_rng(int(bootstrap["seed"]))
    sampled = np.empty(int(bootstrap["samples"]), dtype=np.float64)
    for index in range(sampled.size):
        seed_indices = rng.integers(0, ratios.shape[0], size=ratios.shape[0])
        scenario_indices = rng.integers(
            0,
            ratios.shape[1],
            size=ratios.shape[1],
        )
        sampled[index] = float(np.mean(ratios[np.ix_(seed_indices, scenario_indices)]))
    alpha = (1.0 - float(bootstrap["confidence_level"])) / 2.0
    lower, upper = np.percentile(sampled, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return {
        "seed_count": len(seeds),
        "scenario_count": len(scenario_ids),
        "record_count": len(records),
        "success_count": success_count,
        "failure_count": failure_count,
        "failure_rate": failure_count / len(records),
        "valid_schedule_rate": success_count / len(records),
        "illegal_action_count": illegal_count,
        "illegal_record_count": illegal_record_count,
        "illegal_record_rate": illegal_record_count / len(records),
        "timeout_count": timeout_count,
        "timeout_rate": timeout_count / len(records),
        "mean_ratio": float(np.mean(flat)),
        "seed_mean_ratios": [
            {"seed": int(seed), "mean_ratio": float(seed_means[index])}
            for index, seed in enumerate(seeds)
        ],
        "seed_mean_ratio_std": float(np.std(seed_means)),
        "median_ratio": float(np.percentile(flat, 50)),
        "p95_ratio": float(np.percentile(flat, 95)),
        "win_tie_loss_vs_reference": {
            "win": int(np.sum(flat < 1.0 - tie_tolerance)),
            "tie": int(np.sum(np.abs(flat - 1.0) <= tie_tolerance)),
            "loss": int(np.sum(flat > 1.0 + tie_tolerance)),
        },
        "runtime_ms": {
            "mean": float(np.mean(runtimes)),
            "p50": float(np.percentile(runtimes, 50)),
            "p95": float(np.percentile(runtimes, 95)),
        },
        "hierarchical_bootstrap": {
            "samples": int(bootstrap["samples"]),
            "seed": int(bootstrap["seed"]),
            "confidence_level": float(bootstrap["confidence_level"]),
            "hierarchy": ["seed", "scenario"],
            "mean_ratio_lower": float(lower),
            "mean_ratio_upper": float(upper),
        },
    }


def _aggregate_seed(
    records: list[dict[str, Any]],
    *,
    timeout_error_code: str,
) -> dict[str, Any]:
    ratios = np.asarray(
        [float(row["score_ratio"]) for row in records],
        dtype=np.float64,
    )
    runtimes = np.asarray(
        [float(row["runtime_ms"]) for row in records],
        dtype=np.float64,
    )
    failure_count = sum(row["status"] == "failure" for row in records)
    success_count = len(records) - failure_count
    illegal_action_count = sum(int(row["illegal_action_count"]) for row in records)
    illegal_record_count = sum(int(row["illegal_action_count"]) > 0 for row in records)
    timeout_count = sum(row.get("error_code") == timeout_error_code for row in records)
    return {
        "record_count": len(records),
        "success_count": success_count,
        "failure_count": failure_count,
        "failure_rate": failure_count / len(records),
        "valid_schedule_rate": success_count / len(records),
        "illegal_action_count": illegal_action_count,
        "illegal_record_count": illegal_record_count,
        "timeout_count": timeout_count,
        "mean_ratio": float(np.mean(ratios)),
        "p50_ratio": float(np.percentile(ratios, 50)),
        "p95_ratio": float(np.percentile(ratios, 95)),
        "runtime_mean_ms": float(np.mean(runtimes)),
        "runtime_p50_ms": float(np.percentile(runtimes, 50)),
        "runtime_p95_ms": float(np.percentile(runtimes, 95)),
    }


def _score_matrix(
    records: list[dict[str, Any]],
    seeds: list[int],
    scenario_ids: list[str],
) -> np.ndarray:
    mapped = {(int(row["seed"]), row["scenario_id"]): row for row in records}
    return np.asarray(
        [
            [
                float(mapped[(seed, scenario_id)]["score_ratio"])
                for scenario_id in scenario_ids
            ]
            for seed in seeds
        ],
        dtype=np.float64,
    )


def _paired_policy_comparison(
    primary_records: list[dict[str, Any]],
    primary_seeds: list[int],
    comparator_records: list[dict[str, Any]],
    comparator_seeds: list[int],
    scenario_ids: list[str],
    *,
    bootstrap: dict[str, Any],
    tie_tolerance: float,
) -> dict[str, Any]:
    primary = _score_matrix(primary_records, primary_seeds, scenario_ids)
    comparator = _score_matrix(
        comparator_records,
        comparator_seeds,
        scenario_ids,
    )
    scenario_deltas = np.mean(primary, axis=0) - np.mean(comparator, axis=0)
    rng = np.random.default_rng(int(bootstrap["seed"]))
    sampled = np.empty(int(bootstrap["samples"]), dtype=np.float64)
    for index in range(sampled.size):
        primary_seed_indices = rng.integers(
            0,
            primary.shape[0],
            size=primary.shape[0],
        )
        comparator_seed_indices = rng.integers(
            0,
            comparator.shape[0],
            size=comparator.shape[0],
        )
        scenario_indices = rng.integers(
            0,
            primary.shape[1],
            size=primary.shape[1],
        )
        sampled[index] = float(
            np.mean(primary[np.ix_(primary_seed_indices, scenario_indices)])
            - np.mean(comparator[np.ix_(comparator_seed_indices, scenario_indices)])
        )
    alpha = (1.0 - float(bootstrap["confidence_level"])) / 2.0
    lower, upper = np.percentile(sampled, [100.0 * alpha, 100.0 * (1.0 - alpha)])

    exact_pairs = None
    if set(primary_seeds) == set(comparator_seeds):
        shared_seeds = sorted(primary_seeds)
        exact_deltas = (
            _score_matrix(primary_records, shared_seeds, scenario_ids)
            - _score_matrix(comparator_records, shared_seeds, scenario_ids)
        ).ravel()
        exact_pairs = {
            "pair_count": int(exact_deltas.size),
            "mean_delta": float(np.mean(exact_deltas)),
            "win_tie_loss": {
                "win": int(np.sum(exact_deltas < -tie_tolerance)),
                "tie": int(np.sum(np.abs(exact_deltas) <= tie_tolerance)),
                "loss": int(np.sum(exact_deltas > tie_tolerance)),
            },
        }

    return {
        "direction": "lower_score_ratio_is_better",
        "estimand": "primary_mean_score_ratio_minus_comparator_mean_score_ratio",
        "primary_seed_count": len(primary_seeds),
        "comparator_seed_count": len(comparator_seeds),
        "shared_scenario_count": len(scenario_ids),
        "mean_paired_delta": float(np.mean(scenario_deltas)),
        "scenario_mean_win_tie_loss": {
            "win": int(np.sum(scenario_deltas < -tie_tolerance)),
            "tie": int(np.sum(np.abs(scenario_deltas) <= tie_tolerance)),
            "loss": int(np.sum(scenario_deltas > tie_tolerance)),
        },
        "hierarchical_paired_bootstrap": {
            "samples": int(bootstrap["samples"]),
            "seed": int(bootstrap["seed"]),
            "confidence_level": float(bootstrap["confidence_level"]),
            "seed_sampling": "independent_by_policy",
            "scenario_sampling": "shared_paired_indices",
            "delta_lower": float(lower),
            "delta_upper": float(upper),
            "excludes_zero": bool(upper < 0.0 or lower > 0.0),
            "supports_primary_improvement": bool(upper < -tie_tolerance),
        },
        "all_seed_scenario_pairs": exact_pairs,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_evaluation_report(
    contract_path: str | Path,
    evidence_path: str | Path,
    output_dir: str | Path,
    *,
    test_receipt_path: str | Path | None = None,
) -> Path:
    contract_source = Path(contract_path).resolve()
    evidence_source = Path(evidence_path).resolve()
    destination = Path(output_dir).resolve()
    contract = load_evaluation_contract(contract_source)
    contract_hash = canonical_json_sha256(contract)
    evidence, indexed, scenarios, receipt = _validate_evidence(
        evidence_source,
        contract,
        contract_hash,
        test_receipt_path,
    )
    try:
        if destination.exists():
            if not destination.is_dir():
                _fail("report_output", "$output", "output path must be a directory")
            if any(destination.iterdir()):
                _fail(
                    "report_output",
                    "$output",
                    "refusing to mix report artifacts in a non-empty directory",
                )
        destination.mkdir(parents=True, exist_ok=True)
    except EvaluationReportError:
        raise
    except OSError as error:
        _fail("report_output", "$output", str(error))
    policies, slices = _contract_maps(contract)
    required_slices = contract["modes"][evidence["mode"]]
    slice_reports: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    seed_csv_rows: list[dict[str, Any]] = []
    comparison_csv_rows: list[dict[str, Any]] = []
    for slice_id in required_slices:
        scenario_ids = [item["scenario_id"] for item in scenarios[slice_id]]
        policy_reports: dict[str, dict[str, Any]] = {}
        policy_rows: dict[str, list[dict[str, Any]]] = {}
        for policy_id, policy in policies.items():
            seeds = list(policy["required_seeds"])
            rows = [
                indexed[(slice_id, policy_id, seed, scenario_id)]
                for seed in seeds
                for scenario_id in scenario_ids
            ]
            policy_rows[policy_id] = rows
            for seed in seeds:
                seed_records = [
                    indexed[(slice_id, policy_id, seed, scenario_id)]
                    for scenario_id in scenario_ids
                ]
                seed_metrics = _aggregate_seed(
                    seed_records,
                    timeout_error_code=contract["metrics"]["timeout_error_code"],
                )
                seed_csv_rows.append(
                    {
                        "slice_id": slice_id,
                        "slice_role": slices[slice_id]["role"],
                        "policy": policy_id,
                        "policy_role": policy["role"],
                        "seed": seed,
                        **seed_metrics,
                    }
                )
            metrics = _aggregate_policy(
                rows,
                seeds,
                scenario_ids,
                bootstrap=contract["bootstrap"],
                tie_tolerance=float(contract["tie_tolerance"]),
                timeout_error_code=contract["metrics"]["timeout_error_code"],
            )
            policy_reports[policy_id] = metrics
            csv_rows.append(
                {
                    "slice_id": slice_id,
                    "slice_role": slices[slice_id]["role"],
                    "policy": policy_id,
                    "policy_role": policy["role"],
                    "seed_count": metrics["seed_count"],
                    "scenario_count": metrics["scenario_count"],
                    "record_count": metrics["record_count"],
                    "success_count": metrics["success_count"],
                    "mean_ratio": metrics["mean_ratio"],
                    "seed_mean_ratio_std": metrics["seed_mean_ratio_std"],
                    "p50_ratio": metrics["median_ratio"],
                    "p95_ratio": metrics["p95_ratio"],
                    "failure_rate": metrics["failure_rate"],
                    "valid_schedule_rate": metrics["valid_schedule_rate"],
                    "illegal_action_count": metrics["illegal_action_count"],
                    "illegal_record_rate": metrics["illegal_record_rate"],
                    "timeout_rate": metrics["timeout_rate"],
                    "runtime_p50_ms": metrics["runtime_ms"]["p50"],
                    "runtime_p95_ms": metrics["runtime_ms"]["p95"],
                    "win": metrics["win_tie_loss_vs_reference"]["win"],
                    "tie": metrics["win_tie_loss_vs_reference"]["tie"],
                    "loss": metrics["win_tie_loss_vs_reference"]["loss"],
                    "bootstrap_ci_low": metrics["hierarchical_bootstrap"][
                        "mean_ratio_lower"
                    ],
                    "bootstrap_ci_high": metrics["hierarchical_bootstrap"][
                        "mean_ratio_upper"
                    ],
                }
            )
        primary_id = contract["primary_policy"]
        comparisons: dict[str, dict[str, Any]] = {}
        for comparator_id, comparator_policy in policies.items():
            if comparator_id == primary_id:
                continue
            comparison = _paired_policy_comparison(
                policy_rows[primary_id],
                list(policies[primary_id]["required_seeds"]),
                policy_rows[comparator_id],
                list(comparator_policy["required_seeds"]),
                scenario_ids,
                bootstrap=contract["bootstrap"],
                tie_tolerance=float(contract["tie_tolerance"]),
            )
            comparisons[comparator_id] = comparison
            paired_bootstrap = comparison["hierarchical_paired_bootstrap"]
            outcomes = comparison["scenario_mean_win_tie_loss"]
            exact_pairs = comparison["all_seed_scenario_pairs"]
            comparison_csv_rows.append(
                {
                    "slice_id": slice_id,
                    "primary_policy": primary_id,
                    "comparator_policy": comparator_id,
                    "comparator_role": comparator_policy["role"],
                    "primary_seed_count": comparison["primary_seed_count"],
                    "comparator_seed_count": comparison["comparator_seed_count"],
                    "shared_scenario_count": comparison["shared_scenario_count"],
                    "mean_paired_delta": comparison["mean_paired_delta"],
                    "win": outcomes["win"],
                    "tie": outcomes["tie"],
                    "loss": outcomes["loss"],
                    "bootstrap_delta_low": paired_bootstrap["delta_lower"],
                    "bootstrap_delta_high": paired_bootstrap["delta_upper"],
                    "supports_primary_improvement": paired_bootstrap[
                        "supports_primary_improvement"
                    ],
                    "exact_seed_scenario_pair_count": (
                        exact_pairs["pair_count"] if exact_pairs is not None else ""
                    ),
                }
            )
        slice_reports.append(
            {
                "slice_id": slice_id,
                "role": slices[slice_id]["role"],
                "definition": slices[slice_id]["definition"],
                "scenario_count": len(scenario_ids),
                "scenario_set_sha256": evidence["slice_manifests"][slice_id][
                    "scenario_set_sha256"
                ],
                "policies": policy_reports,
                "primary_comparisons": comparisons,
            }
        )

    primary = contract["primary_policy"]
    primary_metrics = [item["policies"][primary] for item in slice_reports]
    primary_zero_failures = all(
        item["failure_count"] == 0 and item["illegal_action_count"] == 0
        for item in primary_metrics
    )
    primary_below_reference = all(item["mean_ratio"] < 1.0 for item in primary_metrics)
    final_mode = evidence["mode"] == "final_test"
    release_publishable = bool(
        final_mode
        and receipt is not None
        and evidence["code"]["working_tree_dirty"] is False
        and primary_zero_failures
        and primary_below_reference
    )
    report = {
        "format_version": 1,
        "mode": "p1_b02_aggregated_evaluation_report",
        "report_scope": evidence["mode"],
        "contract": {
            "id": contract["contract_id"],
            "sha256": contract_hash,
        },
        "evidence": {
            "name": evidence_source.name,
            "sha256": file_sha256(evidence_source),
            "records_sha256": evidence["records_sha256"],
            "code": evidence["code"],
            "test_accessed": evidence["test_accessed"],
        },
        "scoring": {
            "primary_policy": primary,
            "reference_policy": contract["reference_policy"],
            "failure_penalty_ratio": contract["failure_penalty_ratio"],
            "tie_tolerance": contract["tie_tolerance"],
            "failures_remain_in_denominator": True,
            "runtime_ms": contract["metrics"]["runtime_ms"],
            "runtime_quantiles": contract["metrics"]["runtime_quantiles"],
            "timeout_error_code": contract["metrics"]["timeout_error_code"],
        },
        "slices": slice_reports,
        "gate": {
            "primary_zero_failures_and_illegal_actions": primary_zero_failures,
            "primary_mean_ratio_below_reference_on_every_reported_slice": (
                primary_below_reference
            ),
            "release_publishable": release_publishable,
        },
        "test_gate_receipt": (
            {
                "name": Path(test_receipt_path).name,
                "sha256": file_sha256(test_receipt_path),
                "status": receipt["status"],
            }
            if receipt is not None and test_receipt_path is not None
            else None
        ),
        "claim_boundary": (
            "development/OOD evidence is not a final generalization claim"
            if not final_mode
            else "final public-test report accepted under the one-time gate receipt"
        ),
    }
    report_path = destination / "evaluation_report.json"
    csv_path = destination / "evaluation_per_slice.csv"
    seed_csv_path = destination / "evaluation_per_seed.csv"
    comparison_csv_path = destination / "evaluation_primary_comparisons.csv"
    try:
        _write_json(report_path, report)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
        with seed_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(seed_csv_rows[0]))
            writer.writeheader()
            writer.writerows(seed_csv_rows)
        with comparison_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(comparison_csv_rows[0]),
            )
            writer.writeheader()
            writer.writerows(comparison_csv_rows)
        manifest = {
            "format_version": 1,
            "mode": "p1_b02_evaluation_report_manifest",
            "inputs": {
                "contract": {
                    "name": contract_source.name,
                    "file_sha256": file_sha256(contract_source),
                    "canonical_sha256": contract_hash,
                },
                "evidence": {
                    "name": evidence_source.name,
                    "sha256": file_sha256(evidence_source),
                },
                "test_gate_receipt": (
                    {
                        "name": Path(test_receipt_path).name,
                        "sha256": file_sha256(test_receipt_path),
                    }
                    if test_receipt_path is not None
                    else None
                ),
            },
            "artifacts": {
                report_path.name: {
                    "bytes": report_path.stat().st_size,
                    "sha256": file_sha256(report_path),
                },
                csv_path.name: {
                    "bytes": csv_path.stat().st_size,
                    "sha256": file_sha256(csv_path),
                },
                seed_csv_path.name: {
                    "bytes": seed_csv_path.stat().st_size,
                    "sha256": file_sha256(seed_csv_path),
                },
                comparison_csv_path.name: {
                    "bytes": comparison_csv_path.stat().st_size,
                    "sha256": file_sha256(comparison_csv_path),
                },
            },
            "test_accessed": evidence["test_accessed"],
        }
        _write_json(destination / "evaluation_report_manifest.json", manifest)
    except (OSError, TypeError, ValueError) as error:
        _fail("report_output_write", "$output", str(error))
    return report_path
