from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from trisched.env import ScheduleEntry, ScheduleResult, run_policy, validate_schedule
from trisched.policies import HeftPolicy, compute_upward_ranks
from trisched.scenario import Scenario


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "golden" / "heft_cases.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
CASES: list[dict[str, Any]] = FIXTURE["cases"]
TOLERANCE = float(FIXTURE["tolerance"])


def test_golden_fixture_inventory_is_complete() -> None:
    assert FIXTURE["format_version"] == 1
    assert len(CASES) == 10
    assert len({case["id"] for case in CASES}) == 10
    assert all(case["id"] == case["scenario"]["id"] for case in CASES)
    assert all(case["coverage"] for case in CASES)


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_heft_matches_hand_derived_golden_case(case: dict[str, Any]) -> None:
    scenario = Scenario.from_dict(case["scenario"])
    expected = case["expected"]

    ranks = compute_upward_ranks(scenario)
    np.testing.assert_allclose(
        ranks,
        np.asarray(expected["upward_ranks"], dtype=np.float64),
        rtol=0.0,
        atol=TOLERANCE,
    )

    result = run_policy(scenario, HeftPolicy())
    validate_schedule(scenario, result)
    assert [entry.task_id for entry in result.entries] == expected["decision_order"]
    assert result.makespan == pytest.approx(expected["makespan"], abs=TOLERANCE)

    assert len(result.entries) == len(expected["schedule"])
    for actual, golden in zip(result.entries, expected["schedule"]):
        assert actual.task_id == golden["task_id"]
        assert actual.resource_id == golden["resource_id"]
        assert actual.start == pytest.approx(golden["start"], abs=TOLERANCE)
        assert actual.finish == pytest.approx(golden["finish"], abs=TOLERANCE)


def test_case_3_manual_alternative_is_valid_and_beats_heft() -> None:
    case = next(case for case in CASES if case["id"] == "G03_diamond_heft_myopia")
    scenario = Scenario.from_dict(case["scenario"])
    expected = case["expected"]
    alternative = expected["manual_alternative"]
    entries = tuple(ScheduleEntry(**entry) for entry in alternative["schedule"])
    result = ScheduleResult(
        scenario_id=scenario.id,
        policy_name="manual_alternative",
        entries=entries,
        makespan=float(alternative["makespan"]),
    )

    validate_schedule(scenario, result)
    ratio = result.makespan / float(expected["makespan"])
    assert ratio == pytest.approx(alternative["ratio_vs_heft"], abs=TOLERANCE)
    assert ratio < 1.0
