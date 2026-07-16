from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from trisched.env import run_policy
from trisched.exact import ExactSolverLimitError, solve_exact_schedule
from trisched.oracle import validate_schedule_independent
from trisched.policies import CpopPolicy, HeftPolicy, analyze_cpop
from trisched.scenario import Scenario, generate_scenario


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "golden"
HEFT_FIXTURE = json.loads(
    (FIXTURE_DIR / "heft_cases.json").read_text(encoding="utf-8")
)
OPTIMAL_FIXTURE = json.loads(
    (FIXTURE_DIR / "optimal_cases.json").read_text(encoding="utf-8")
)
SCENARIOS = {
    case["id"]: Scenario.from_dict(case["scenario"])
    for case in HEFT_FIXTURE["cases"]
}
HEFT_MAKESPANS = {
    case["id"]: float(case["expected"]["makespan"])
    for case in HEFT_FIXTURE["cases"]
}
CASES: list[dict[str, Any]] = OPTIMAL_FIXTURE["cases"]
TOLERANCE = float(OPTIMAL_FIXTURE["tolerance"])


def test_optimal_fixture_covers_every_golden_scenario_once() -> None:
    assert OPTIMAL_FIXTURE["format_version"] == 1
    assert OPTIMAL_FIXTURE["source_fixture"] == "heft_cases.json"
    assert len(CASES) == 10
    assert len({case["id"] for case in CASES}) == 10
    assert {case["id"] for case in CASES} == set(SCENARIOS)


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_cpop_and_exact_solver_match_frozen_small_graph_results(
    case: dict[str, Any],
) -> None:
    scenario = SCENARIOS[case["id"]]
    expected_cpop = case["cpop"]
    analysis = analyze_cpop(scenario)

    np.testing.assert_allclose(
        analysis.downward_ranks,
        np.asarray(expected_cpop["downward_ranks"], dtype=np.float64),
        rtol=0.0,
        atol=TOLERANCE,
    )
    np.testing.assert_allclose(
        analysis.priorities,
        np.asarray(expected_cpop["priorities"], dtype=np.float64),
        rtol=0.0,
        atol=TOLERANCE,
    )
    assert analysis.critical_path == tuple(expected_cpop["critical_path"])
    assert analysis.critical_resource == expected_cpop["critical_resource"]

    cpop = run_policy(scenario, CpopPolicy())
    validate_schedule_independent(scenario, cpop)
    assert [entry.task_id for entry in cpop.entries] == expected_cpop[
        "decision_order"
    ]
    assert [entry.resource_id for entry in cpop.entries] == expected_cpop[
        "resources"
    ]
    np.testing.assert_allclose(
        [entry.start for entry in cpop.entries],
        expected_cpop["starts"],
        rtol=0.0,
        atol=TOLERANCE,
    )
    np.testing.assert_allclose(
        [entry.finish for entry in cpop.entries],
        expected_cpop["finishes"],
        rtol=0.0,
        atol=TOLERANCE,
    )
    assert cpop.makespan == pytest.approx(
        expected_cpop["makespan"], abs=TOLERANCE
    )
    by_task = {entry.task_id: entry for entry in cpop.entries}
    assert all(
        by_task[task_id].resource_id == analysis.critical_resource
        for task_id in analysis.critical_path
    )

    exact = solve_exact_schedule(scenario)
    validate_schedule_independent(scenario, exact)
    assert exact.proven_optimal is True
    assert exact.explored_states > 0
    assert exact.pruned_states > 0
    assert exact.analytical_lower_bound == pytest.approx(
        case["analytical_lower_bound"], abs=TOLERANCE
    )
    assert exact.makespan == pytest.approx(
        case["optimal_makespan"], abs=TOLERANCE
    )
    assert exact.analytical_lower_bound <= exact.makespan + TOLERANCE
    assert exact.makespan <= HEFT_MAKESPANS[case["id"]] + TOLERANCE
    assert exact.makespan <= cpop.makespan + TOLERANCE


def test_exact_solver_rejects_instances_above_explicit_task_limit() -> None:
    scenario = generate_scenario(seed=91, task_count=9, resource_count=2)
    with pytest.raises(ExactSolverLimitError, match="at most 8 tasks"):
        solve_exact_schedule(scenario)


def test_exact_solver_reports_state_budget_exhaustion() -> None:
    scenario = SCENARIOS["G03_diamond_heft_myopia"]
    with pytest.raises(ExactSolverLimitError, match="max_states=1"):
        solve_exact_schedule(scenario, max_states=1)


def test_cpop_is_available_through_the_common_policy_runner() -> None:
    scenario = SCENARIOS["G06_asymmetric_fast_download"]
    result = run_policy(scenario, CpopPolicy())
    heft = run_policy(scenario, HeftPolicy())
    assert result.policy_name == "cpop"
    assert result.makespan > heft.makespan
