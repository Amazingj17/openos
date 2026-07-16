from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from trisched.env import ScheduleResult, run_policy
from trisched.oracle import (
    IndependentValidationError,
    independent_heft_schedule,
    independent_upward_ranks,
    validate_schedule_independent,
)
from trisched.policies import (
    CpopPolicy,
    GreedyEarliestFinishPolicy,
    HeftPolicy,
    RandomPolicy,
    compute_upward_ranks,
)
from trisched.scenario import Scenario, generate_scenario


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "golden" / "heft_cases.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
CASES: list[dict[str, Any]] = FIXTURE["cases"]
TOLERANCE = float(FIXTURE["tolerance"])


def _assert_schedule_matches_golden(result: Any, expected: dict[str, Any]) -> None:
    assert [entry.task_id for entry in result.entries] == expected["decision_order"]
    assert result.makespan == pytest.approx(expected["makespan"], abs=TOLERANCE)
    assert len(result.entries) == len(expected["schedule"])
    for actual, golden in zip(result.entries, expected["schedule"]):
        assert actual.task_id == golden["task_id"]
        assert actual.resource_id == golden["resource_id"]
        assert actual.start == pytest.approx(golden["start"], abs=TOLERANCE)
        assert actual.finish == pytest.approx(golden["finish"], abs=TOLERANCE)


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_golden_production_and_independent_oracle_agree(
    case: dict[str, Any],
) -> None:
    """Compare hand derivation, production HEFT, and the independent oracle."""
    scenario = Scenario.from_dict(case["scenario"])
    expected = case["expected"]
    expected_ranks = np.asarray(expected["upward_ranks"], dtype=np.float64)

    np.testing.assert_allclose(
        compute_upward_ranks(scenario),
        expected_ranks,
        rtol=0.0,
        atol=TOLERANCE,
    )
    np.testing.assert_allclose(
        independent_upward_ranks(scenario),
        expected_ranks,
        rtol=0.0,
        atol=TOLERANCE,
    )

    production = run_policy(scenario, HeftPolicy())
    oracle = independent_heft_schedule(scenario)
    _assert_schedule_matches_golden(production, expected)
    _assert_schedule_matches_golden(oracle, expected)


def test_oracle_does_not_call_production_scenario_timing_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = Scenario.from_dict(CASES[2]["scenario"])
    production = run_policy(scenario, HeftPolicy())

    def fail_if_called(*args: Any, **kwargs: Any) -> float:
        raise AssertionError("production timing helper was called")

    monkeypatch.setattr(Scenario, "execution_time", fail_if_called)
    monkeypatch.setattr(Scenario, "communication_time", fail_if_called)

    oracle = independent_heft_schedule(scenario)
    report = validate_schedule_independent(scenario, production)
    assert oracle.makespan == pytest.approx(production.makespan)
    assert report.task_count == scenario.task_count


@pytest.mark.parametrize("case_index", range(40), ids=lambda value: f"dag-{value:02d}")
def test_random_dag_schedules_satisfy_independent_properties(
    case_index: int,
) -> None:
    scenario = generate_scenario(
        seed=10_000 + case_index,
        task_count=5 + case_index % 8,
        resource_count=1 + case_index % 4,
        edge_probability=(case_index % 6) / 10,
        scenario_id=f"property-{case_index:02d}",
    )
    policies = (
        HeftPolicy(),
        CpopPolicy(),
        GreedyEarliestFinishPolicy(),
        RandomPolicy(seed=70_000 + case_index),
    )

    for policy in policies:
        result = run_policy(scenario, policy)
        report = validate_schedule_independent(scenario, result)
        assert report.task_count == scenario.task_count
        assert report.resource_count == scenario.resource_count
        assert report.makespan == pytest.approx(result.makespan)


def _corrupt_result(kind: str) -> tuple[Scenario, ScheduleResult]:
    case = CASES[1] if kind == "resource_overlap" else CASES[0]
    scenario = Scenario.from_dict(case["scenario"])
    base = run_policy(scenario, HeftPolicy())
    entries = list(base.entries)
    makespan = base.makespan

    if kind == "missing_task":
        entries.pop()
    elif kind == "duplicate_task":
        entries[-1] = entries[0]
    elif kind == "unknown_task":
        entries[-1] = replace(entries[-1], task_id=scenario.task_count)
    elif kind == "unknown_resource":
        entries[-1] = replace(entries[-1], resource_id=scenario.resource_count)
    elif kind == "invalid_timestamp":
        entries[-1] = replace(entries[-1], finish=entries[-1].start)
    elif kind == "non_finite_timestamp":
        entries[-1] = replace(entries[-1], start=float("nan"))
    elif kind == "wrong_duration":
        entries[-1] = replace(entries[-1], finish=entries[-1].finish + 0.25)
    elif kind == "dependency_violation":
        entries[-1] = replace(entries[-1], start=0.5, finish=1.5)
    elif kind == "resource_overlap":
        entries[-1] = replace(
            entries[-1], resource_id=1, start=1.0, finish=3.0
        )
        makespan = 3.0
    elif kind == "inconsistent_makespan":
        makespan += 1.0
    else:  # pragma: no cover - protects the test helper itself
        raise AssertionError(f"unknown fault kind: {kind}")

    return scenario, ScheduleResult(
        scenario_id=base.scenario_id,
        policy_name=f"fault-{kind}",
        entries=tuple(entries),
        makespan=makespan,
    )


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("missing_task", "does not contain every task"),
        ("duplicate_task", "appears more than once"),
        ("unknown_task", "unknown task"),
        ("unknown_resource", "unknown resource"),
        ("invalid_timestamp", "invalid timestamps"),
        ("non_finite_timestamp", "non-finite timestamp"),
        ("wrong_duration", "wrong execution duration"),
        ("dependency_violation", "dependency 0->1 is violated"),
        ("resource_overlap", "tasks overlap on resource 1"),
        ("inconsistent_makespan", "reported makespan is inconsistent"),
    ],
)
def test_independent_validator_rejects_injected_faults(
    kind: str,
    message: str,
) -> None:
    scenario, corrupted = _corrupt_result(kind)
    with pytest.raises(IndependentValidationError, match=message):
        validate_schedule_independent(scenario, corrupted)
