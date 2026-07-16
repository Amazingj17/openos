from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Resource:
    id: int
    name: str
    kind: str
    speed: float


@dataclass(frozen=True)
class Task:
    id: int
    workload: float


@dataclass(frozen=True)
class Edge:
    source: int
    target: int
    data: float


@dataclass(frozen=True)
class Scenario:
    """A static heterogeneous DAG scheduling instance."""

    id: str
    seed: int
    tasks: tuple[Task, ...]
    resources: tuple[Resource, ...]
    edges: tuple[Edge, ...]
    bandwidth: tuple[tuple[float, ...], ...]
    latency: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        self.validate()

    @property
    def task_count(self) -> int:
        return len(self.tasks)

    @property
    def resource_count(self) -> int:
        return len(self.resources)

    def predecessors(self) -> tuple[tuple[int, ...], ...]:
        pred: list[list[int]] = [[] for _ in self.tasks]
        for edge in self.edges:
            pred[edge.target].append(edge.source)
        return tuple(tuple(sorted(items)) for items in pred)

    def successors(self) -> tuple[tuple[int, ...], ...]:
        succ: list[list[int]] = [[] for _ in self.tasks]
        for edge in self.edges:
            succ[edge.source].append(edge.target)
        return tuple(tuple(sorted(items)) for items in succ)

    def edge_data(self) -> dict[tuple[int, int], float]:
        return {(edge.source, edge.target): edge.data for edge in self.edges}

    def execution_time(self, task_id: int, resource_id: int) -> float:
        return self.tasks[task_id].workload / self.resources[resource_id].speed

    def communication_time(
        self, source: int, target: int, source_resource: int, target_resource: int
    ) -> float:
        if source_resource == target_resource:
            return 0.0
        data = self.edge_data()[(source, target)]
        return (
            self.latency[source_resource][target_resource]
            + data / self.bandwidth[source_resource][target_resource]
        )

    def validate(self) -> None:
        n = len(self.tasks)
        m = len(self.resources)
        if n == 0 or m == 0:
            raise ValueError("scenario requires at least one task and one resource")
        if [task.id for task in self.tasks] != list(range(n)):
            raise ValueError("task ids must be contiguous and start at zero")
        if [resource.id for resource in self.resources] != list(range(m)):
            raise ValueError("resource ids must be contiguous and start at zero")
        if any(task.workload <= 0 for task in self.tasks):
            raise ValueError("task workloads must be positive")
        if any(resource.speed <= 0 for resource in self.resources):
            raise ValueError("resource speeds must be positive")
        if len(self.bandwidth) != m or any(len(row) != m for row in self.bandwidth):
            raise ValueError("bandwidth must be a resource_count square matrix")
        if len(self.latency) != m or any(len(row) != m for row in self.latency):
            raise ValueError("latency must be a resource_count square matrix")
        if any(self.bandwidth[i][j] <= 0 for i in range(m) for j in range(m)):
            raise ValueError("all bandwidth entries must be positive")
        if any(self.latency[i][j] < 0 for i in range(m) for j in range(m)):
            raise ValueError("latency entries cannot be negative")
        seen: set[tuple[int, int]] = set()
        indegree = [0] * n
        succ: list[list[int]] = [[] for _ in range(n)]
        for edge in self.edges:
            if not (0 <= edge.source < n and 0 <= edge.target < n):
                raise ValueError("edge endpoint outside task range")
            if edge.source == edge.target or edge.data < 0:
                raise ValueError("invalid dependency edge")
            key = (edge.source, edge.target)
            if key in seen:
                raise ValueError("duplicate dependency edge")
            seen.add(key)
            indegree[edge.target] += 1
            succ[edge.source].append(edge.target)
        queue = [i for i, degree in enumerate(indegree) if degree == 0]
        visited = 0
        while queue:
            node = queue.pop()
            visited += 1
            for child in succ[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if visited != n:
            raise ValueError("task graph must be acyclic")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "seed": self.seed,
            "tasks": [asdict(task) for task in self.tasks],
            "resources": [asdict(resource) for resource in self.resources],
            "edges": [asdict(edge) for edge in self.edges],
            "bandwidth": [list(row) for row in self.bandwidth],
            "latency": [list(row) for row in self.latency],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scenario":
        return cls(
            id=str(data["id"]),
            seed=int(data.get("seed", 0)),
            tasks=tuple(Task(**item) for item in data["tasks"]),
            resources=tuple(Resource(**item) for item in data["resources"]),
            edges=tuple(Edge(**item) for item in data["edges"]),
            bandwidth=tuple(tuple(float(x) for x in row) for row in data["bandwidth"]),
            latency=tuple(tuple(float(x) for x in row) for row in data["latency"]),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "Scenario":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def content_hash(self) -> str:
        content = self.to_dict()
        # Split leakage checks must not be defeated by a different display id or seed.
        content.pop("id", None)
        content.pop("seed", None)
        payload = json.dumps(
            content, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _resource_kind(index: int, count: int) -> str:
    if count == 1:
        return "cloud"
    position = index / max(count - 1, 1)
    if position < 0.34:
        return "device"
    if position < 0.67:
        return "edge"
    return "cloud"


def _link_profile(source_kind: str, target_kind: str) -> tuple[float, float]:
    if source_kind == target_kind:
        return 18.0, 0.05
    pair = frozenset((source_kind, target_kind))
    if pair == frozenset(("device", "edge")):
        return 7.0, 0.18
    if pair == frozenset(("edge", "cloud")):
        return 12.0, 0.28
    return 3.5, 0.65


def generate_scenario(
    seed: int,
    task_count: int = 12,
    resource_count: int = 3,
    edge_probability: float = 0.18,
    scenario_id: str | None = None,
) -> Scenario:
    if task_count < 2:
        raise ValueError("task_count must be at least two")
    if resource_count < 1:
        raise ValueError("resource_count must be positive")
    if not 0 <= edge_probability <= 1:
        raise ValueError("edge_probability must be between zero and one")
    rng = np.random.default_rng(seed)
    tasks = tuple(
        Task(id=i, workload=float(rng.uniform(2.0, 12.0))) for i in range(task_count)
    )
    resources: list[Resource] = []
    speed_base = {"device": 1.0, "edge": 2.4, "cloud": 4.8}
    kind_counts = {"device": 0, "edge": 0, "cloud": 0}
    for i in range(resource_count):
        kind = _resource_kind(i, resource_count)
        suffix = kind_counts[kind]
        kind_counts[kind] += 1
        resources.append(
            Resource(
                id=i,
                name=f"{kind}-{suffix}",
                kind=kind,
                speed=float(speed_base[kind] * rng.uniform(0.85, 1.15)),
            )
        )

    edge_pairs: set[tuple[int, int]] = set()
    # Give every non-root task one predecessor while keeping multiple tasks ready.
    for target in range(1, task_count):
        lower = max(0, target - max(3, task_count // 3))
        source = int(rng.integers(lower, target))
        edge_pairs.add((source, target))
    for source in range(task_count):
        for target in range(source + 1, task_count):
            if rng.random() < edge_probability:
                edge_pairs.add((source, target))
    edges = tuple(
        Edge(source=s, target=t, data=float(rng.uniform(0.5, 8.0)))
        for s, t in sorted(edge_pairs)
    )

    bandwidth: list[list[float]] = []
    latency: list[list[float]] = []
    for source in resources:
        bw_row: list[float] = []
        lat_row: list[float] = []
        for target in resources:
            if source.id == target.id:
                bw_row.append(1e9)
                lat_row.append(0.0)
            else:
                bw, lat = _link_profile(source.kind, target.kind)
                bw_row.append(float(bw * rng.uniform(0.88, 1.12)))
                lat_row.append(float(lat * rng.uniform(0.9, 1.1)))
        bandwidth.append(bw_row)
        latency.append(lat_row)
    return Scenario(
        id=scenario_id or f"scenario-{seed}",
        seed=seed,
        tasks=tasks,
        resources=tuple(resources),
        edges=edges,
        bandwidth=tuple(tuple(row) for row in bandwidth),
        latency=tuple(tuple(row) for row in latency),
    )


def generate_dataset(
    count: int,
    seed: int,
    task_range: tuple[int, int] = (8, 16),
    resource_count: int = 3,
    edge_probability: float = 0.18,
    prefix: str = "dataset",
) -> list[Scenario]:
    if count <= 0:
        raise ValueError("dataset count must be positive")
    low, high = task_range
    if low < 2 or high < low:
        raise ValueError("invalid task_range")
    chooser = np.random.default_rng(seed)
    scenarios: list[Scenario] = []
    for index in range(count):
        scenario_seed = seed + (index + 1) * 9973
        task_count = int(chooser.integers(low, high + 1))
        scenarios.append(
            generate_scenario(
                seed=scenario_seed,
                task_count=task_count,
                resource_count=resource_count,
                edge_probability=edge_probability,
                scenario_id=f"{prefix}-{index:04d}",
            )
        )
    return scenarios
