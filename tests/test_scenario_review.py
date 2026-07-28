from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from trisched.scenario import Scenario, ScenarioValidationError


# This payload and every mutation below were written independently for the A review.
# In particular, this module must not read tests/fixtures/invalid/scenario_cases.json.
BASE_PAYLOAD: dict[str, Any] = {
    "id": "a-review",
    "seed": 20260716,
    "tasks": [
        {"id": 0, "workload": 2.0},
        {"id": 1, "workload": 3.0},
        {"id": 2, "workload": 1.0},
    ],
    "resources": [
        {"id": 0, "name": "edge-a", "kind": "edge", "speed": 2.0},
        {"id": 1, "name": "cloud-a", "kind": "cloud", "speed": 4.0},
    ],
    "edges": [
        {"source": 0, "target": 1, "data": 1.0},
        {"source": 1, "target": 2, "data": 1.0},
    ],
    "bandwidth": [[100.0, 8.0], [6.0, 100.0]],
    "latency": [[0.0, 0.2], [0.3, 0.0]],
}


REVIEW_CASES = (
    ("missing_field", "missing_field", "$.latency"),
    ("nested_unknown_field", "unknown_field", "$.tasks[0].review_note"),
    ("boolean_task_id", "type_error", "$.tasks[0].id"),
    ("overflow_number", "non_finite", "$.tasks[0].workload"),
    ("ragged_matrix", "matrix_shape", "$.bandwidth"),
    ("duplicate_edge", "duplicate_edge", "$.edges[2]"),
    ("cycle", "cycle", "$.edges"),
)


def _invalid_payload(case_id: str) -> dict[str, Any]:
    payload = copy.deepcopy(BASE_PAYLOAD)
    if case_id == "missing_field":
        del payload["latency"]
    elif case_id == "nested_unknown_field":
        payload["tasks"][0]["review_note"] = "must be rejected"
    elif case_id == "boolean_task_id":
        payload["tasks"][0]["id"] = True
    elif case_id == "overflow_number":
        payload["tasks"][0]["workload"] = float("inf")
    elif case_id == "ragged_matrix":
        payload["bandwidth"][1] = [6.0]
    elif case_id == "duplicate_edge":
        payload["edges"].append({"source": 0, "target": 1, "data": 9.0})
    elif case_id == "cycle":
        payload["edges"].append({"source": 2, "target": 0, "data": 1.0})
    else:
        raise AssertionError(f"unknown independent review case: {case_id}")
    return payload


def _write_cli_payload(path: Path, case_id: str) -> None:
    payload = _invalid_payload(case_id)
    if case_id == "overflow_number":
        # JSON itself contains the finite-looking number literal under review.
        # review. Python decodes 1e309 as +inf, which runtime validation must reject.
        payload["tasks"][0]["workload"] = "__OVERFLOW__"
        encoded = json.dumps(payload, ensure_ascii=False).replace(
            '"__OVERFLOW__"', "1e309"
        )
    else:
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    path.write_text(encoded, encoding="utf-8")


def _run_cli(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "trisched",
            "validate-scenario",
            "--input",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("case_id", "expected_code", "expected_path"),
    REVIEW_CASES,
    ids=[case[0] for case in REVIEW_CASES],
)
def test_a_review_api_reports_independently_derived_diagnostics(
    case_id: str,
    expected_code: str,
    expected_path: str,
) -> None:
    with pytest.raises(ScenarioValidationError) as caught:
        Scenario.from_dict(_invalid_payload(case_id))

    error = caught.value
    assert error.code == expected_code
    assert error.path == expected_path
    assert error.to_dict() == {
        "code": expected_code,
        "path": expected_path,
        "message": error.detail,
    }


@pytest.mark.parametrize(
    ("case_id", "expected_code", "expected_path"),
    REVIEW_CASES,
    ids=[case[0] for case in REVIEW_CASES],
)
def test_a_review_cli_returns_two_and_structured_json(
    tmp_path: Path,
    case_id: str,
    expected_code: str,
    expected_path: str,
) -> None:
    path = tmp_path / f"{case_id}.json"
    _write_cli_payload(path, case_id)

    result = _run_cli(path)

    assert result.returncode == 2, result.stderr
    assert result.stderr == ""
    output = json.loads(result.stdout)
    assert output["valid"] is False
    assert output["error"]["code"] == expected_code
    assert output["error"]["path"] == expected_path
    assert isinstance(output["error"]["message"], str)
    assert output["error"]["message"]


def test_a_review_load_and_cli_reject_non_utf8_bytes(tmp_path: Path) -> None:
    path = tmp_path / "gbk.json"
    path.write_bytes('{"id":"场景"}'.encode("gbk"))

    with pytest.raises(ScenarioValidationError) as caught:
        Scenario.load(path)
    assert caught.value.code == "encoding_error"
    assert caught.value.path == "$"

    result = _run_cli(path)
    assert result.returncode == 2, result.stderr
    output = json.loads(result.stdout)
    assert output["valid"] is False
    assert output["error"]["code"] == "encoding_error"
    assert output["error"]["path"] == "$"


def test_a_review_valid_payload_passes_api_and_cli(tmp_path: Path) -> None:
    scenario = Scenario.from_dict(copy.deepcopy(BASE_PAYLOAD))
    path = tmp_path / "valid.json"
    path.write_text(
        json.dumps(BASE_PAYLOAD, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    result = _run_cli(path)

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    output = json.loads(result.stdout)
    assert output == {
        "valid": True,
        "scenario_id": "a-review",
        "task_count": 3,
        "resource_count": 2,
        "content_hash": scenario.content_hash(),
    }
