from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from .scenario import Scenario


EPSILON = 1e-9


class IndependentValidationError(ValueError):
    """Raised when the independent schedule oracle rejects a result."""


@dataclass(frozen=True)
class OracleScheduleEntry:
    task_id: int
    resource_id: int
    start: float
    finish: float


@dataclass(frozen=True)
class OracleScheduleResult:
    scenario_id: str
    policy_name: str
    entries: tuple[OracleScheduleEntry, ...]
    makespan: float


@dataclass(frozen=True)
class IndependentValidationReport:
    scenario_id: str
    task_count: int
    resource_count: int
    makespan: float


def _execution_time(scenario: Scenario, task_id: int, resource_id: int) -> float:
    # Intentionally do not call Scenario.execution_time().
    return (
        float(scenario.tasks[task_id].workload)
        / float(scenario.resources[resource_id].speed)
    )


def _edge_data(scenario: Scenario) -> dict[tuple[int, int], float]:
    return {
        (int(edge.source), int(edge.target)): float(edge.data)
        for edge in scenario.edges
    }


def _communication_time(
    scenario: Scenario,
    edge_data: dict[tuple[int, int], float],
    source: int,
    target: int,
    source_resource: int,
    target_resource: int,
) -> float:
    # Intentionally do not call Scenario.communication_time().
    if source_resource == target_resource:
        return 0.0
    return float(scenario.latency[source_resource][target_resource]) + edge_data[
        (source, target)
    ] / float(scenario.bandwidth[source_resource][target_resource])


def _graph_maps(
    scenario: Scenario,
) -> tuple[list[list[int]], list[list[int]], dict[tuple[int, int], float]]:
    predecessors = [[] for _ in scenario.tasks]
    successors = [[] for _ in scenario.tasks]
    edge_data = _edge_data(scenario)
    for edge in scenario.edges:
        predecessors[edge.target].append(edge.source)
        successors[edge.source].append(edge.target)
    for items in predecessors:
        items.sort()
    for items in successors:
        items.sort()
    return predecessors, successors, edge_data


def _topological_order(successors: list[list[int]]) -> tuple[int, ...]:
    indegree = [0] * len(successors)
    for children in successors:
        for child in children:
            indegree[child] += 1
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
    if len(order) != len(successors):
        raise IndependentValidationError("oracle received a cyclic task graph")
    return tuple(order)


def independent_upward_ranks(scenario: Scenario) -> np.ndarray:
    """Recompute HEFT upward ranks without using the production policy helpers."""
    _, successors, edge_data = _graph_maps(scenario)
    ranks = np.zeros(len(scenario.tasks), dtype=np.float64)
    resource_count = len(scenario.resources)
    for task_id in reversed(_topological_order(successors)):
        average_execution = sum(
            _execution_time(scenario, task_id, resource_id)
            for resource_id in range(resource_count)
        ) / resource_count
        downstream = 0.0
        if successors[task_id]:
            candidates: list[float] = []
            for child in successors[task_id]:
                if resource_count == 1:
                    average_communication = 0.0
                else:
                    cross_costs = [
                        _communication_time(
                            scenario,
                            edge_data,
                            task_id,
                            child,
                            source_resource,
                            target_resource,
                        )
                        for source_resource in range(resource_count)
                        for target_resource in range(resource_count)
                        if source_resource != target_resource
                    ]
                    average_communication = sum(cross_costs) / len(cross_costs)
                candidates.append(average_communication + float(ranks[child]))
            downstream = max(candidates)
        ranks[task_id] = average_execution + downstream
    return ranks


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
        resource_intervals[resource_id], key=lambda item: (item.start, item.finish)
    ):
        if start + duration <= interval.start + EPSILON:
            break
        if start < interval.finish:
            start = interval.finish
    return start, start + duration


def independent_heft_schedule(scenario: Scenario) -> OracleScheduleResult:
    """Build a HEFT schedule through a second, deliberately separate code path."""
    predecessors, _, edge_data = _graph_maps(scenario)
    ranks = independent_upward_ranks(scenario)
    entries: dict[int, OracleScheduleEntry] = {}
    intervals: list[list[OracleScheduleEntry]] = [
        [] for _ in scenario.resources
    ]
    decision_order: list[int] = []
    while len(entries) < len(scenario.tasks):
        ready = [
            task.id
            for task in scenario.tasks
            if task.id not in entries
            and all(parent in entries for parent in predecessors[task.id])
        ]
        if not ready:
            raise IndependentValidationError(
                "oracle found no ready task before the schedule completed"
            )
        task_id = min(ready, key=lambda item: (-float(ranks[item]), item))
        candidates = [
            (
                *_earliest_slot(
                    scenario,
                    edge_data,
                    predecessors,
                    entries,
                    intervals,
                    task_id,
                    resource.id,
                ),
                resource.id,
            )
            for resource in scenario.resources
        ]
        start, finish, resource_id = min(
            candidates, key=lambda item: (item[1], item[2])
        )
        entry = OracleScheduleEntry(task_id, resource_id, start, finish)
        entries[task_id] = entry
        intervals[resource_id].append(entry)
        decision_order.append(task_id)
    ordered = tuple(entries[task_id] for task_id in decision_order)
    result = OracleScheduleResult(
        scenario_id=scenario.id,
        policy_name="independent_heft_oracle",
        entries=ordered,
        makespan=max(entry.finish for entry in ordered),
    )
    validate_schedule_independent(scenario, result)
    return result


def validate_schedule_independent(
    scenario: Scenario,
    result: Any,
    tolerance: float = 1e-7,
) -> IndependentValidationReport:
    """Validate raw schedule fields without calling the production environment."""
    if result.scenario_id != scenario.id:
        raise IndependentValidationError("scenario id does not match")
    entries = tuple(result.entries)
    if len(entries) != len(scenario.tasks):
        raise IndependentValidationError("schedule does not contain every task")
    edge_data = _edge_data(scenario)
    by_task: dict[int, Any] = {}
    by_resource: list[list[Any]] = [[] for _ in scenario.resources]
    for entry in entries:
        if entry.task_id in by_task:
            raise IndependentValidationError(
                f"task {entry.task_id} appears more than once"
            )
        if not 0 <= entry.task_id < len(scenario.tasks):
            raise IndependentValidationError("schedule contains an unknown task")
        if not 0 <= entry.resource_id < len(scenario.resources):
            raise IndependentValidationError("schedule contains an unknown resource")
        if not isfinite(entry.start) or not isfinite(entry.finish):
            raise IndependentValidationError("schedule contains a non-finite timestamp")
        if entry.start < -tolerance or entry.finish <= entry.start:
            raise IndependentValidationError(
                f"task {entry.task_id} has invalid timestamps"
            )
        expected_duration = _execution_time(
            scenario, entry.task_id, entry.resource_id
        )
        if abs((entry.finish - entry.start) - expected_duration) > tolerance:
            raise IndependentValidationError(
                f"task {entry.task_id} has the wrong execution duration"
            )
        by_task[entry.task_id] = entry
        by_resource[entry.resource_id].append(entry)
    if set(by_task) != set(range(len(scenario.tasks))):
        raise IndependentValidationError("schedule task set is incomplete")

    for edge in scenario.edges:
        parent = by_task[edge.source]
        child = by_task[edge.target]
        transfer = _communication_time(
            scenario,
            edge_data,
            edge.source,
            edge.target,
            parent.resource_id,
            child.resource_id,
        )
        if child.start + tolerance < parent.finish + transfer:
            raise IndependentValidationError(
                f"dependency {edge.source}->{edge.target} is violated"
            )
    for resource_id, resource_entries in enumerate(by_resource):
        ordered = sorted(resource_entries, key=lambda item: (item.start, item.finish))
        for previous, current in zip(ordered, ordered[1:]):
            if current.start + tolerance < previous.finish:
                raise IndependentValidationError(
                    f"tasks overlap on resource {resource_id}: "
                    f"{previous.task_id} and {current.task_id}"
                )
    expected_makespan = max(entry.finish for entry in entries)
    if not isfinite(result.makespan):
        raise IndependentValidationError("reported makespan is not finite")
    if abs(float(result.makespan) - expected_makespan) > tolerance:
        raise IndependentValidationError("reported makespan is inconsistent")
    return IndependentValidationReport(
        scenario_id=scenario.id,
        task_count=len(entries),
        resource_count=len(scenario.resources),
        makespan=expected_makespan,
    )
