from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from trisched.cli import main
from trisched.scenario import (
    Resource,
    Scenario,
    ScenarioValidationError,
    Task,
    generate_scenario,
)


ROOT = Path(__file__).parents[1]
SCHEMA_PATH = ROOT / "schemas" / "scenario.schema.json"
INVALID_FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "invalid" / "scenario_cases.json"
)
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
INVALID_FIXTURE = json.loads(INVALID_FIXTURE_PATH.read_text(encoding="utf-8"))
CASES: list[dict[str, Any]] = INVALID_FIXTURE["cases"]


def test_schema_declares_closed_draft_2020_scenario_contract() -> None:
    assert SCHEMA["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert SCHEMA["type"] == "object"
    assert SCHEMA["additionalProperties"] is False
    assert set(SCHEMA["required"]) == {
        "id",
        "tasks",
        "resources",
        "edges",
        "bandwidth",
        "latency",
    }
    assert SCHEMA["$defs"]["task"]["additionalProperties"] is False
    assert SCHEMA["$defs"]["resource"]["properties"]["kind"]["enum"] == [
        "device",
        "edge",
        "cloud",
    ]
    assert "contiguous IDs" in SCHEMA["$comment"]


def test_invalid_fixture_inventory_is_complete() -> None:
    assert INVALID_FIXTURE["format_version"] == 1
    assert len(CASES) == 12
    assert len({case["id"] for case in CASES}) == len(CASES)
    assert all(set(case["expected"]) == {"code", "path"} for case in CASES)


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_fixed_invalid_scenarios_have_stable_diagnostics(
    case: dict[str, Any],
) -> None:
    with pytest.raises(ScenarioValidationError) as caught:
        Scenario.from_dict(case["payload"])
    error = caught.value
    assert error.code == case["expected"]["code"]
    assert error.path == case["expected"]["path"]
    assert error.to_dict() == {
        "code": error.code,
        "path": error.path,
        "message": error.detail,
    }
    assert error.code in str(error)
    assert error.path in str(error)


def test_nested_unknown_field_reports_exact_path() -> None:
    payload = generate_scenario(seed=7, task_count=2, resource_count=1).to_dict()
    payload["tasks"][0]["unexpected"] = True
    with pytest.raises(ScenarioValidationError) as caught:
        Scenario.from_dict(payload)
    assert caught.value.code == "unknown_field"
    assert caught.value.path == "$.tasks[0].unexpected"


@pytest.mark.parametrize("constant", ("NaN", "Infinity", "-Infinity"))
def test_load_rejects_non_standard_json_constants(
    tmp_path: Path,
    constant: str,
) -> None:
    path = tmp_path / "non-finite.json"
    path.write_text(constant, encoding="utf-8")
    with pytest.raises(ScenarioValidationError) as caught:
        Scenario.load(path)
    assert caught.value.code == "non_finite"
    assert caught.value.path == "$"


def test_load_reports_json_syntax_location(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text('{"id":', encoding="utf-8")
    with pytest.raises(ScenarioValidationError) as caught:
        Scenario.load(path)
    assert caught.value.code == "json_syntax"
    assert caught.value.path == "$"
    assert "line 1" in caught.value.detail


def test_load_accepts_optional_utf8_bom(tmp_path: Path) -> None:
    scenario = generate_scenario(seed=31, task_count=3, resource_count=2)
    path = tmp_path / "bom.json"
    payload = json.dumps(scenario.to_dict(), ensure_ascii=False).encode("utf-8")
    path.write_bytes(b"\xef\xbb\xbf" + payload)
    assert Scenario.load(path) == scenario


def test_load_reports_non_utf8_encoding(tmp_path: Path) -> None:
    path = tmp_path / "invalid-encoding.json"
    path.write_bytes(b"\xff\xfe\x00\x00")
    with pytest.raises(ScenarioValidationError) as caught:
        Scenario.load(path)
    assert caught.value.code == "encoding_error"
    assert caught.value.path == "$"


def test_direct_constructor_rejects_non_finite_workload() -> None:
    with pytest.raises(ScenarioValidationError) as caught:
        Scenario(
            id="direct-nan",
            seed=0,
            tasks=(Task(0, float("nan")),),
            resources=(Resource(0, "cloud-0", "cloud", 1.0),),
            edges=(),
            bandwidth=((1e9,),),
            latency=((0.0,),),
        )
    assert caught.value.code == "non_finite"
    assert caught.value.path == "$.tasks[0].workload"


def test_save_load_round_trip_preserves_hash(tmp_path: Path) -> None:
    scenario = generate_scenario(seed=41, task_count=5, resource_count=3)
    path = tmp_path / "scenario.json"
    scenario.save(path)
    loaded = Scenario.load(path)
    assert loaded == scenario
    assert loaded.content_hash() == scenario.content_hash()


def test_validate_scenario_cli_emits_valid_payload(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scenario = generate_scenario(seed=51, task_count=3, resource_count=2)
    path = tmp_path / "valid.json"
    scenario.save(path)
    assert main(["validate-scenario", "--input", str(path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["scenario_id"] == scenario.id
    assert payload["content_hash"] == scenario.content_hash()


def test_validate_scenario_cli_emits_error_and_exit_two(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text('{"id":', encoding="utf-8")
    assert main(["validate-scenario", "--input", str(path)]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is False
    assert payload["error"]["code"] == "json_syntax"
    assert payload["error"]["path"] == "$"
