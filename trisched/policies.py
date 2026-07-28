from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .env import HeterogeneousDagEnv
from .scenario import Scenario


class Scheduler(Protocol):
    name: str

    def reset(self, scenario: Scenario) -> None: ...

    def select_action(self, env: HeterogeneousDagEnv) -> tuple[int, int]: ...


def topological_order(scenario: Scenario) -> tuple[int, ...]:
    predecessors = scenario.predecessors()
    successors = scenario.successors()
    indegree = [len(items) for items in predecessors]
    ready = [task_id for task_id, degree in enumerate(indegree) if degree == 0]
    order: list[int] = []
    while ready:
        node = min(ready)
        ready.remove(node)
        order.append(node)
        for child in successors[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(order) != scenario.task_count:
        raise ValueError("cannot rank a cyclic task graph")
    return tuple(order)


def average_communication_cost(scenario: Scenario, source: int, target: int) -> float:
    if scenario.resource_count == 1:
        return 0.0
    costs = [
        scenario.communication_time(source, target, source_resource, target_resource)
        for source_resource in range(scenario.resource_count)
        for target_resource in range(scenario.resource_count)
        if source_resource != target_resource
    ]
    return float(np.mean(costs))


def average_execution_cost(scenario: Scenario, task_id: int) -> float:
    return float(
        np.mean(
            [
                scenario.execution_time(task_id, resource_id)
                for resource_id in range(scenario.resource_count)
            ]
        )
    )


def compute_upward_ranks(scenario: Scenario) -> np.ndarray:
    """Compute the HEFT upward rank for every task."""

    successors = scenario.successors()
    ranks = np.zeros(scenario.task_count, dtype=np.float64)
    for task_id in reversed(topological_order(scenario)):
        average_execution = average_execution_cost(scenario, task_id)
        if successors[task_id]:
            downstream = max(
                average_communication_cost(scenario, task_id, child) + ranks[child]
                for child in successors[task_id]
            )
        else:
            downstream = 0.0
        ranks[task_id] = average_execution + downstream
    return ranks


def compute_downward_ranks(scenario: Scenario) -> np.ndarray:
    """Compute CPOP downward ranks from roots toward each task."""

    predecessors = scenario.predecessors()
    ranks = np.zeros(scenario.task_count, dtype=np.float64)
    for task_id in topological_order(scenario):
        if predecessors[task_id]:
            ranks[task_id] = max(
                ranks[parent]
                + average_execution_cost(scenario, parent)
                + average_communication_cost(scenario, parent, task_id)
                for parent in predecessors[task_id]
            )
    return ranks


@dataclass(frozen=True)
class CpopAnalysis:
    upward_ranks: np.ndarray
    downward_ranks: np.ndarray
    priorities: np.ndarray
    critical_path: tuple[int, ...]
    critical_resource: int


def analyze_cpop(scenario: Scenario) -> CpopAnalysis:
    """Build deterministic CPOP priorities, critical chain, and processor."""

    predecessors = scenario.predecessors()
    successors = scenario.successors()
    upward = compute_upward_ranks(scenario)
    downward = compute_downward_ranks(scenario)
    priorities = upward + downward

    roots = [
        task_id
        for task_id in range(scenario.task_count)
        if not predecessors[task_id]
    ]
    current = min(roots, key=lambda task_id: (-float(upward[task_id]), task_id))
    critical_path = [current]
    while successors[current]:
        current = min(
            successors[current],
            key=lambda child: (
                -(
                    average_communication_cost(scenario, current, child)
                    + float(upward[child])
                ),
                child,
            ),
        )
        critical_path.append(current)

    critical_resource = min(
        range(scenario.resource_count),
        key=lambda resource_id: (
            sum(
                scenario.execution_time(task_id, resource_id)
                for task_id in critical_path
            ),
            resource_id,
        ),
    )
    return CpopAnalysis(
        upward_ranks=upward,
        downward_ranks=downward,
        priorities=priorities,
        critical_path=tuple(critical_path),
        critical_resource=critical_resource,
    )


class HeftPolicy:
    name = "heft"

    def __init__(self) -> None:
        self.ranks: np.ndarray | None = None

    def reset(self, scenario: Scenario) -> None:
        self.ranks = compute_upward_ranks(scenario)

    def select_action(self, env: HeterogeneousDagEnv) -> tuple[int, int]:
        if self.ranks is None:
            self.reset(env.scenario)
        assert self.ranks is not None
        ready = env.ready_tasks()
        task_id = min(ready, key=lambda item: (-self.ranks[item], item))
        resource_id = min(
            range(env.scenario.resource_count),
            key=lambda item: (env.earliest_slot(task_id, item)[1], item),
        )
        return task_id, resource_id


class CpopPolicy:
    """Critical Path on a Processor with deterministic tie-breaking."""

    name = "cpop"

    def __init__(self) -> None:
        self.analysis: CpopAnalysis | None = None
        self._critical_tasks: frozenset[int] = frozenset()

    def reset(self, scenario: Scenario) -> None:
        self.analysis = analyze_cpop(scenario)
        self._critical_tasks = frozenset(self.analysis.critical_path)

    def select_action(self, env: HeterogeneousDagEnv) -> tuple[int, int]:
        if self.analysis is None:
            self.reset(env.scenario)
        assert self.analysis is not None
        ready = env.ready_tasks()
        task_id = min(
            ready,
            key=lambda item: (-float(self.analysis.priorities[item]), item),
        )
        if task_id in self._critical_tasks:
            resource_id = self.analysis.critical_resource
        else:
            resource_id = min(
                range(env.scenario.resource_count),
                key=lambda item: (env.earliest_slot(task_id, item)[1], item),
            )
        return task_id, resource_id


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.base_seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, scenario: Scenario) -> None:
        self.rng = np.random.default_rng(self.base_seed + scenario.seed)

    def select_action(self, env: HeterogeneousDagEnv) -> tuple[int, int]:
        candidates = env.candidate_actions()
        index = int(self.rng.integers(0, len(candidates)))
        return candidates[index]


class GreedyEarliestFinishPolicy:
    name = "greedy_eft"

    def reset(self, scenario: Scenario) -> None:
        return None

    def select_action(self, env: HeterogeneousDagEnv) -> tuple[int, int]:
        return min(
            env.candidate_actions(),
            key=lambda action: (
                env.earliest_slot(action[0], action[1])[1],
                action[0],
                action[1],
            ),
        )
