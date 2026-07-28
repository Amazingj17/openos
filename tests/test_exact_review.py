from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

import pytest

from trisched.env import run_policy
from trisched.exact import solve_exact_schedule
from trisched.policies import CpopPolicy, analyze_cpop
from trisched.scenario import Edge, Resource, Scenario, Task


TOLERANCE = 1e-9


@dataclass(frozen=True)
class ReviewOptimum:
    assignment: tuple[int, ...]
    starts: tuple[float, ...]
    makespan: float
    feasible_resource_orders: int


def _review_scenarios() -> dict[str, Scenario]:
    two_resources_fast_cloud = (
        Resource(0, "edge-0", "edge", 1.0),
        Resource(1, "cloud-0", "cloud", 2.0),
    )
    common_join = {
        "seed": 0,
        "tasks": (Task(0, 8.0), Task(1, 4.0), Task(2, 2.0)),
        "resources": two_resources_fast_cloud,
        "edges": (Edge(0, 2, 2.0), Edge(1, 2, 2.0)),
        "latency": ((0.0, 0.0), (0.0, 0.0)),
    }
    return {
        "G03_diamond_heft_myopia": Scenario(
            id="G03_diamond_heft_myopia",
            seed=0,
            tasks=(Task(0, 4.0), Task(1, 4.0), Task(2, 8.0), Task(3, 2.0)),
            resources=(
                Resource(0, "edge-0", "edge", 2.0),
                Resource(1, "cloud-0", "cloud", 4.0),
            ),
            edges=(
                Edge(0, 1, 2.0),
                Edge(0, 2, 4.0),
                Edge(1, 3, 2.0),
                Edge(2, 3, 2.0),
            ),
            bandwidth=((1e9, 2.0), (4.0, 1e9)),
            latency=((0.0, 0.5), (0.25, 0.0)),
        ),
        "G05_insertion_gap": Scenario(
            id="G05_insertion_gap",
            seed=0,
            tasks=(
                Task(0, 40.0),
                Task(1, 1.0),
                Task(2, 20.0),
                Task(3, 10.0),
            ),
            resources=(
                Resource(0, "device-0", "device", 1.0),
                Resource(1, "cloud-0", "cloud", 10.0),
            ),
            edges=(Edge(1, 2, 10.0),),
            bandwidth=((1e9, 1.0), (1.0, 1e9)),
            latency=((0.0, 0.0), (0.0, 0.0)),
        ),
        "G06_asymmetric_fast_download": Scenario(
            id="G06_asymmetric_fast_download",
            bandwidth=((1e9, 1.0), (4.0, 1e9)),
            **common_join,
        ),
        "G07_asymmetric_fast_upload": Scenario(
            id="G07_asymmetric_fast_upload",
            bandwidth=((1e9, 4.0), (1.0, 1e9)),
            **common_join,
        ),
    }


SCENARIOS = _review_scenarios()
EXPECTED = {
    "G03_diamond_heft_myopia": {
        "optimum": 4.5,
        "assignment": (1, 1, 1, 1),
        "cpop_path": (0, 2, 3),
        "cpop_resource": 1,
        "cpop_makespan": 5.75,
    },
    "G05_insertion_gap": {
        "optimum": 7.1,
        "assignment": (1, 1, 1, 1),
        "cpop_path": (0,),
        "cpop_resource": 1,
        "cpop_makespan": 13.0,
    },
    "G06_asymmetric_fast_download": {
        "optimum": 6.5,
        "assignment": (1, 0, 0),
        "cpop_path": (0, 2),
        "cpop_resource": 1,
        "cpop_makespan": 7.0,
    },
    "G07_asymmetric_fast_upload": {
        "optimum": 5.5,
        "assignment": (1, 0, 1),
        "cpop_path": (0, 2),
        "cpop_resource": 1,
        "cpop_makespan": 5.5,
    },
}


def _timing_data(
    scenario: Scenario, assignment: tuple[int, ...]
) -> tuple[list[float], dict[tuple[int, int], float]]:
    durations = [
        float(task.workload)
        / float(scenario.resources[assignment[task.id]].speed)
        for task in scenario.tasks
    ]
    edge_data = {(edge.source, edge.target): float(edge.data) for edge in scenario.edges}
    transfers: dict[tuple[int, int], float] = {}
    for edge in scenario.edges:
        source_resource = assignment[edge.source]
        target_resource = assignment[edge.target]
        if source_resource == target_resource:
            transfers[(edge.source, edge.target)] = 0.0
        else:
            transfers[(edge.source, edge.target)] = float(
                scenario.latency[source_resource][target_resource]
            ) + edge_data[(edge.source, edge.target)] / float(
                scenario.bandwidth[source_resource][target_resource]
            )
    return durations, transfers


def _constraint_graph_schedule(
    scenario: Scenario,
    assignment: tuple[int, ...],
    resource_orders: tuple[tuple[int, ...], ...],
) -> tuple[tuple[float, ...], float] | None:
    """Solve a fixed assignment/order through DAG longest paths."""

    durations, transfers = _timing_data(scenario, assignment)
    successors: list[list[tuple[int, float]]] = [[] for _ in scenario.tasks]
    indegree = [0] * scenario.task_count
    for edge in scenario.edges:
        lag = durations[edge.source] + transfers[(edge.source, edge.target)]
        successors[edge.source].append((edge.target, lag))
        indegree[edge.target] += 1
    for order in resource_orders:
        for previous, current in zip(order, order[1:]):
            successors[previous].append((current, durations[previous]))
            indegree[current] += 1

    ready = [task_id for task_id, degree in enumerate(indegree) if degree == 0]
    starts = [0.0] * scenario.task_count
    visited = 0
    while ready:
        task_id = min(ready)
        ready.remove(task_id)
        visited += 1
        for child, lag in successors[task_id]:
            starts[child] = max(starts[child], starts[task_id] + lag)
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if visited != scenario.task_count:
        return None
    makespan = max(
        starts[task_id] + durations[task_id]
        for task_id in range(scenario.task_count)
    )
    return tuple(starts), makespan


def _enumerate_assignment_and_resource_orders(scenario: Scenario) -> ReviewOptimum:
    """A's reference method; structurally separate from B's ready-task search."""

    best_assignment: tuple[int, ...] | None = None
    best_starts: tuple[float, ...] | None = None
    best_makespan = float("inf")
    feasible_orders = 0
    for assignment in product(
        range(scenario.resource_count), repeat=scenario.task_count
    ):
        tasks_by_resource = [
            tuple(
                task_id
                for task_id, resource_id in enumerate(assignment)
                if resource_id == candidate_resource
            )
            for candidate_resource in range(scenario.resource_count)
        ]
        order_options = [tuple(permutations(tasks)) for tasks in tasks_by_resource]
        for resource_orders in product(*order_options):
            scheduled = _constraint_graph_schedule(
                scenario, assignment, resource_orders
            )
            if scheduled is None:
                continue
            feasible_orders += 1
            starts, makespan = scheduled
            if makespan < best_makespan - TOLERANCE:
                best_assignment = tuple(assignment)
                best_starts = starts
                best_makespan = makespan
    assert best_assignment is not None
    assert best_starts is not None
    return ReviewOptimum(
        assignment=best_assignment,
        starts=best_starts,
        makespan=best_makespan,
        feasible_resource_orders=feasible_orders,
    )


@pytest.mark.parametrize("scenario_id", tuple(SCENARIOS))
def test_a_resource_order_oracle_reviews_b_exact_and_cpop(
    scenario_id: str,
) -> None:
    scenario = SCENARIOS[scenario_id]
    expected = EXPECTED[scenario_id]

    review = _enumerate_assignment_and_resource_orders(scenario)
    assert review.feasible_resource_orders > 0
    assert review.makespan == pytest.approx(expected["optimum"], abs=TOLERANCE)
    assert review.assignment == expected["assignment"]

    exact = solve_exact_schedule(scenario)
    assert exact.proven_optimal is True
    assert exact.makespan == pytest.approx(review.makespan, abs=TOLERANCE)

    analysis = analyze_cpop(scenario)
    assert analysis.critical_path == expected["cpop_path"]
    assert analysis.critical_resource == expected["cpop_resource"]
    cpop = run_policy(scenario, CpopPolicy())
    assert cpop.makespan == pytest.approx(
        expected["cpop_makespan"], abs=TOLERANCE
    )
