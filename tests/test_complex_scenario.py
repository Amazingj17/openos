from __future__ import annotations

import pytest

from trisched.env import HeterogeneousDagEnv, run_policy
from trisched.policies import HeftPolicy
from trisched.scenario import Edge, Resource, Scenario, Task, generate_complex_scenario


def constrained_scenario() -> Scenario:
    return Scenario(
        id="constraints",
        seed=1,
        tasks=(
            Task(0, 8.0, cpu_cores_required=2, memory_required=4.0),
            Task(
                1,
                10.0,
                cpu_cores_required=8,
                memory_required=16.0,
                accelerator_required=True,
                required_features=("gpu",),
                task_type="gpu",
            ),
        ),
        resources=(
            Resource(0, "device", "device", 1.0, 4, 8.0),
            Resource(1, "edge-gpu", "edge", 4.0, 16, 32.0, True, ("gpu",)),
            Resource(2, "cloud", "cloud", 8.0, 64, 128.0, True, ("gpu",)),
        ),
        edges=(Edge(0, 1, 2.0),),
        bandwidth=((1e9, 10.0, 4.0), (10.0, 1e9, 20.0), (4.0, 20.0, 1e9)),
        latency=((0.0, 0.1, 0.3), (0.1, 0.0, 0.2), (0.3, 0.2, 0.0)),
        execution_times=((8.0, 2.5, 1.5), (20.0, 3.0, 1.0)),
    )


def test_capability_constraints_are_part_of_the_legal_action_mask() -> None:
    scenario = constrained_scenario()
    env = HeterogeneousDagEnv(scenario)
    assert env.candidate_actions() == ((0, 0), (0, 1), (0, 2))
    env.step(0, 0)
    assert env.candidate_actions() == ((1, 1), (1, 2))
    with pytest.raises(ValueError, match="incompatible"):
        env.earliest_slot(1, 0)


def test_explicit_execution_matrix_and_heft_remain_valid() -> None:
    scenario = constrained_scenario()
    assert scenario.execution_time(1, 2) == 1.0
    result = run_policy(scenario, HeftPolicy())
    assert len(result.entries) == 2
    assert next(item for item in result.entries if item.task_id == 1).resource_id == 2


def test_complex_generator_is_reproducible_and_stays_in_three_tiers() -> None:
    first = generate_complex_scenario(7, task_count=30, resource_count=6)
    second = generate_complex_scenario(7, task_count=30, resource_count=6)
    assert first.to_dict() == second.to_dict()
    assert {resource.kind for resource in first.resources} == {
        "device",
        "edge",
        "cloud",
    }
    assert first.execution_times is not None
    assert any(task.accelerator_required for task in first.tasks)
    assert all(first.compatible_resources(task.id) for task in first.tasks)


def test_new_fields_round_trip_without_changing_legacy_defaults() -> None:
    scenario = constrained_scenario()
    assert Scenario.from_dict(scenario.to_dict()) == scenario
    legacy = Scenario.from_dict(
        {
            "id": "legacy",
            "tasks": [{"id": 0, "workload": 1.0}],
            "resources": [
                {"id": 0, "name": "cloud", "kind": "cloud", "speed": 1.0}
            ],
            "edges": [],
            "bandwidth": [[1e9]],
            "latency": [[0.0]],
        }
    )
    assert legacy.tasks[0].cpu_cores_required == 1
    assert legacy.resources[0].cpu_cores == 1
    assert "cpu_cores" not in legacy.to_dict()["resources"][0]

