from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .oracle import validate_schedule_independent
from .scenario import Scenario


EPSILON = 1e-9


class IllegalActionError(ValueError):
    """Typed policy-contract failure for an action outside the legal mask."""

    def __init__(self, policy_name: str, task_id: Any, resource_id: Any) -> None:
        self.policy_name = policy_name
        self.task_id = task_id
        self.resource_id = resource_id
        super().__init__(
            f"policy {policy_name} returned illegal action ({task_id}, {resource_id})"
        )


@dataclass(frozen=True)
class ScheduleEntry:
    task_id: int
    resource_id: int
    start: float
    finish: float

    @property
    def duration(self) -> float:
        return self.finish - self.start


@dataclass(frozen=True)
class ScheduleResult:
    scenario_id: str
    policy_name: str
    entries: tuple[ScheduleEntry, ...]
    makespan: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "policy_name": self.policy_name,
            "makespan": self.makespan,
            "entries": [asdict(entry) for entry in self.entries],
        }


class HeterogeneousDagEnv:
    """Deterministic list-scheduling environment with insertion-based slots."""

    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.predecessors = scenario.predecessors()
        self.successors = scenario.successors()
        self.entries: dict[int, ScheduleEntry] = {}
        self.resource_intervals: list[list[ScheduleEntry]] = [
            [] for _ in scenario.resources
        ]
        self.decision_order: list[int] = []
        self._ready_cache: tuple[int, ...] | None = None

    @property
    def done(self) -> bool:
        return len(self.entries) == self.scenario.task_count

    @property
    def progress(self) -> float:
        return len(self.entries) / self.scenario.task_count

    def ready_tasks(self) -> tuple[int, ...]:
        if self._ready_cache is None:
            self._ready_cache = tuple(
                task.id
                for task in self.scenario.tasks
                if task.id not in self.entries
                and all(parent in self.entries for parent in self.predecessors[task.id])
            )
        return self._ready_cache

    def candidate_actions(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (task_id, resource.id)
            for task_id in self.ready_tasks()
            for resource in self.scenario.resources
        )

    def resource_ready_time(self, resource_id: int) -> float:
        intervals = self.resource_intervals[resource_id]
        return max((entry.finish for entry in intervals), default=0.0)

    def dependency_ready_time(self, task_id: int, resource_id: int) -> float:
        ready = 0.0
        for parent_id in self.predecessors[task_id]:
            parent = self.entries[parent_id]
            transfer = self.scenario.communication_time(
                parent_id,
                task_id,
                parent.resource_id,
                resource_id,
            )
            ready = max(ready, parent.finish + transfer)
        return ready

    def earliest_slot(self, task_id: int, resource_id: int) -> tuple[float, float]:
        if task_id in self.entries:
            raise ValueError(f"task {task_id} is already scheduled")
        if task_id not in self.ready_tasks():
            raise ValueError(f"task {task_id} is not ready")
        if not 0 <= resource_id < self.scenario.resource_count:
            raise ValueError(f"resource {resource_id} does not exist")
        duration = self.scenario.execution_time(task_id, resource_id)
        start = self.dependency_ready_time(task_id, resource_id)
        # step() keeps every resource timeline sorted.
        for interval in self.resource_intervals[resource_id]:
            if start + duration <= interval.start + EPSILON:
                break
            if start < interval.finish:
                start = interval.finish
        return start, start + duration

    def step(self, task_id: int, resource_id: int) -> ScheduleEntry:
        start, finish = self.earliest_slot(task_id, resource_id)
        entry = ScheduleEntry(task_id, resource_id, start, finish)
        self.entries[task_id] = entry
        self._ready_cache = None
        self.resource_intervals[resource_id].append(entry)
        self.resource_intervals[resource_id].sort(
            key=lambda item: (item.start, item.finish, item.task_id)
        )
        self.decision_order.append(task_id)
        return entry

    def result(self, policy_name: str) -> ScheduleResult:
        if not self.done:
            raise ValueError("cannot build a result before every task is scheduled")
        entries = tuple(self.entries[task_id] for task_id in self.decision_order)
        makespan = max(entry.finish for entry in entries)
        return ScheduleResult(self.scenario.id, policy_name, entries, makespan)


def validate_schedule(
    scenario: Scenario, result: ScheduleResult, tolerance: float = 1e-7
) -> None:
    if result.scenario_id != scenario.id:
        raise ValueError("schedule scenario id does not match")
    if len(result.entries) != scenario.task_count:
        raise ValueError("schedule does not contain every task")
    by_task: dict[int, ScheduleEntry] = {}
    by_resource: list[list[ScheduleEntry]] = [[] for _ in scenario.resources]
    for entry in result.entries:
        if entry.task_id in by_task:
            raise ValueError(f"task {entry.task_id} appears more than once")
        if not 0 <= entry.task_id < scenario.task_count:
            raise ValueError("schedule contains an unknown task")
        if not 0 <= entry.resource_id < scenario.resource_count:
            raise ValueError("schedule contains an unknown resource")
        if entry.start < -tolerance or entry.finish <= entry.start:
            raise ValueError(f"task {entry.task_id} has invalid timestamps")
        expected_duration = scenario.execution_time(entry.task_id, entry.resource_id)
        if abs(entry.duration - expected_duration) > tolerance:
            raise ValueError(f"task {entry.task_id} has the wrong execution duration")
        by_task[entry.task_id] = entry
        by_resource[entry.resource_id].append(entry)
    if set(by_task) != set(range(scenario.task_count)):
        raise ValueError("schedule task set is incomplete")

    for edge in scenario.edges:
        parent = by_task[edge.source]
        child = by_task[edge.target]
        transfer = scenario.communication_time(
            edge.source,
            edge.target,
            parent.resource_id,
            child.resource_id,
        )
        if child.start + tolerance < parent.finish + transfer:
            raise ValueError(f"dependency {edge.source}->{edge.target} is violated")
    for resource_id, intervals in enumerate(by_resource):
        ordered = sorted(intervals, key=lambda item: (item.start, item.finish))
        for previous, current in zip(ordered, ordered[1:]):
            if current.start + tolerance < previous.finish:
                raise ValueError(
                    f"tasks overlap on resource {resource_id}: "
                    f"{previous.task_id} and {current.task_id}"
                )
    expected_makespan = max(entry.finish for entry in result.entries)
    if abs(result.makespan - expected_makespan) > tolerance:
        raise ValueError("reported makespan is inconsistent with task completion times")


def run_policy(scenario: Scenario, policy: Any) -> ScheduleResult:
    env = HeterogeneousDagEnv(scenario)
    reset = getattr(policy, "reset", None)
    if reset is not None:
        reset(scenario)
    while not env.done:
        candidates = env.candidate_actions()
        if not candidates:
            raise RuntimeError(
                "no legal action is available before the episode is done"
            )
        task_id, resource_id = policy.select_action(env)
        if (task_id, resource_id) not in candidates:
            raise IllegalActionError(policy.name, task_id, resource_id)
        env.step(task_id, resource_id)
    result = env.result(policy.name)
    validate_schedule(scenario, result)
    validate_schedule_independent(scenario, result)
    return result
