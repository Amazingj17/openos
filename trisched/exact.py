from __future__ import annotations

from dataclasses import dataclass

from .oracle import (
    OracleScheduleEntry,
    independent_heft_schedule,
    validate_schedule_independent,
)
from .scenario import Scenario


EPSILON = 1e-9


class ExactSolverLimitError(RuntimeError):
    """Raised when an instance exceeds the explicit exact-search limits."""


@dataclass(frozen=True)
class ExactScheduleResult:
    scenario_id: str
    policy_name: str
    entries: tuple[OracleScheduleEntry, ...]
    makespan: float
    analytical_lower_bound: float
    explored_states: int
    pruned_states: int
    proven_optimal: bool


def _graph_maps(scenario: Scenario) -> tuple[list[list[int]], list[list[int]]]:
    predecessors = [[] for _ in scenario.tasks]
    successors = [[] for _ in scenario.tasks]
    for edge in scenario.edges:
        predecessors[edge.target].append(edge.source)
        successors[edge.source].append(edge.target)
    for items in predecessors:
        items.sort()
    for items in successors:
        items.sort()
    return predecessors, successors


def _topological_order(
    predecessors: list[list[int]], successors: list[list[int]]
) -> tuple[int, ...]:
    indegree = [len(items) for items in predecessors]
    ready = [task_id for task_id, degree in enumerate(indegree) if degree == 0]
    order: list[int] = []
    while ready:
        task_id = min(ready)
        ready.remove(task_id)
        order.append(task_id)
        for child in successors[task_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    return tuple(order)


def analytical_makespan_lower_bound(scenario: Scenario) -> float:
    """Return a communication-free workload/critical-chain lower bound."""

    predecessors, successors = _graph_maps(scenario)
    fastest_speed = max(float(resource.speed) for resource in scenario.resources)
    fastest_durations = [
        float(task.workload) / fastest_speed for task in scenario.tasks
    ]
    chain_finish = [0.0] * len(scenario.tasks)
    for task_id in _topological_order(predecessors, successors):
        chain_finish[task_id] = fastest_durations[task_id] + max(
            (chain_finish[parent] for parent in predecessors[task_id]),
            default=0.0,
        )
    critical_chain_bound = max(chain_finish)
    capacity_bound = sum(float(task.workload) for task in scenario.tasks) / sum(
        float(resource.speed) for resource in scenario.resources
    )
    return max(critical_chain_bound, capacity_bound)


def _execution_time(scenario: Scenario, task_id: int, resource_id: int) -> float:
    return (
        float(scenario.tasks[task_id].workload)
        / float(scenario.resources[resource_id].speed)
    )


def _communication_time(
    scenario: Scenario,
    edge_data: dict[tuple[int, int], float],
    source: int,
    target: int,
    source_resource: int,
    target_resource: int,
) -> float:
    if source_resource == target_resource:
        return 0.0
    return float(scenario.latency[source_resource][target_resource]) + edge_data[
        (source, target)
    ] / float(scenario.bandwidth[source_resource][target_resource])


def _earliest_slot(
    scenario: Scenario,
    edge_data: dict[tuple[int, int], float],
    predecessors: list[list[int]],
    entries: dict[int, OracleScheduleEntry],
    resource_intervals: list[list[OracleScheduleEntry]],
    task_id: int,
    resource_id: int,
) -> tuple[float, float]:
    dependency_ready = 0.0
    for parent_id in predecessors[task_id]:
        parent = entries[parent_id]
        dependency_ready = max(
            dependency_ready,
            parent.finish
            + _communication_time(
                scenario,
                edge_data,
                parent_id,
                task_id,
                parent.resource_id,
                resource_id,
            ),
        )
    duration = _execution_time(scenario, task_id, resource_id)
    start = dependency_ready
    for interval in sorted(
        resource_intervals[resource_id],
        key=lambda item: (item.start, item.finish, item.task_id),
    ):
        if start + duration <= interval.start + EPSILON:
            break
        if start < interval.finish:
            start = interval.finish
    return start, start + duration


def solve_exact_schedule(
    scenario: Scenario,
    *,
    max_tasks: int = 8,
    max_states: int = 2_000_000,
) -> ExactScheduleResult:
    """Prove a small instance optimum by exhaustive active-schedule search.

    The search enumerates every ready-task choice and resource assignment. For a
    fixed choice sequence it places each task in its earliest feasible insertion
    slot. Deliberate idle time cannot improve makespan in the current model, so
    this covers an optimal active schedule. The current best makespan is a sound
    branch-and-bound cutoff because adding tasks cannot reduce it.
    """

    if max_tasks <= 0 or max_states <= 0:
        raise ValueError("exact solver limits must be positive")
    if scenario.task_count > max_tasks:
        raise ExactSolverLimitError(
            f"exact solver supports at most {max_tasks} tasks; "
            f"received {scenario.task_count}"
        )

    predecessors, _ = _graph_maps(scenario)
    edge_data = {
        (int(edge.source), int(edge.target)): float(edge.data)
        for edge in scenario.edges
    }
    initial = independent_heft_schedule(scenario)
    best_makespan = float(initial.makespan)
    best_entries = {entry.task_id: entry for entry in initial.entries}
    best_order = [entry.task_id for entry in initial.entries]

    entries: dict[int, OracleScheduleEntry] = {}
    resource_intervals: list[list[OracleScheduleEntry]] = [
        [] for _ in scenario.resources
    ]
    decision_order: list[int] = []
    explored_states = 0
    pruned_states = 0

    def search(current_makespan: float) -> None:
        nonlocal best_entries
        nonlocal best_makespan
        nonlocal best_order
        nonlocal explored_states
        nonlocal pruned_states

        explored_states += 1
        if explored_states > max_states:
            raise ExactSolverLimitError(
                f"exact solver exceeded max_states={max_states}"
            )
        if current_makespan >= best_makespan - EPSILON:
            pruned_states += 1
            return
        if len(entries) == scenario.task_count:
            best_makespan = current_makespan
            best_entries = dict(entries)
            best_order = list(decision_order)
            return

        ready = [
            task.id
            for task in scenario.tasks
            if task.id not in entries
            and all(parent in entries for parent in predecessors[task.id])
        ]
        candidates: list[tuple[float, int, int, float]] = []
        for task_id in ready:
            for resource in scenario.resources:
                start, finish = _earliest_slot(
                    scenario,
                    edge_data,
                    predecessors,
                    entries,
                    resource_intervals,
                    task_id,
                    resource.id,
                )
                candidates.append((finish, task_id, resource.id, start))
        candidates.sort()

        for finish, task_id, resource_id, start in candidates:
            entry = OracleScheduleEntry(task_id, resource_id, start, finish)
            entries[task_id] = entry
            resource_intervals[resource_id].append(entry)
            decision_order.append(task_id)
            search(max(current_makespan, finish))
            decision_order.pop()
            resource_intervals[resource_id].pop()
            del entries[task_id]

    search(0.0)
    ordered_entries = tuple(best_entries[task_id] for task_id in best_order)
    result = ExactScheduleResult(
        scenario_id=scenario.id,
        policy_name="exact_branch_and_bound",
        entries=ordered_entries,
        makespan=best_makespan,
        analytical_lower_bound=analytical_makespan_lower_bound(scenario),
        explored_states=explored_states,
        pruned_states=pruned_states,
        proven_optimal=True,
    )
    validate_schedule_independent(scenario, result)
    return result
