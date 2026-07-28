from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np


class ScenarioValidationError(ValueError):
    """A stable, machine-readable Scenario input or invariant error."""

    def __init__(self, message: str, *, code: str, path: str) -> None:
        self.code = code
        self.path = path
        self.detail = message
        super().__init__(f"{code} at {path}: {message}")

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path, "message": self.detail}


def _fail(message: str, *, code: str, path: str) -> None:
    raise ScenarioValidationError(message, code=code, path=path)


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail("expected a JSON object", code="type_error", path=path)
    if any(not isinstance(key, str) for key in value):
        _fail("object keys must be strings", code="type_error", path=path)
    return value


def _sequence(value: Any, path: str) -> list[Any] | tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        _fail("expected a JSON array", code="type_error", path=path)
    return value


def _keys(
    value: dict[str, Any],
    *,
    required: set[str],
    optional: set[str] | frozenset[str] = frozenset(),
    path: str,
) -> None:
    missing = sorted(required - set(value))
    if missing:
        field = missing[0]
        _fail(
            f"missing required field '{field}'",
            code="missing_field",
            path=f"{path}.{field}",
        )
    unknown = sorted(set(value) - required - optional)
    if unknown:
        field = unknown[0]
        _fail(
            f"unknown field '{field}'",
            code="unknown_field",
            path=f"{path}.{field}",
        )


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _fail("expected an integer", code="type_error", path=path)
    return int(value)


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _fail("expected a number", code="type_error", path=path)
    result = float(value)
    if not isfinite(result):
        _fail("number must be finite", code="non_finite", path=path)
    return result


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str):
        _fail("expected a string", code="type_error", path=path)
    if not value:
        _fail("string must not be empty", code="value_error", path=path)
    return value


def _matrix(value: Any, path: str) -> tuple[tuple[float, ...], ...]:
    rows = _sequence(value, path)
    return tuple(
        tuple(
            _number(item, f"{path}[{row_index}][{column_index}]")
            for column_index, item in enumerate(_sequence(row, f"{path}[{row_index}]"))
        )
        for row_index, row in enumerate(rows)
    )


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
        if not isinstance(self.id, str) or not self.id:
            _fail(
                "scenario id must be a non-empty string",
                code="value_error",
                path="$.id",
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            _fail("scenario seed must be an integer", code="type_error", path="$.seed")
        if n == 0:
            _fail(
                "scenario requires at least one task",
                code="value_error",
                path="$.tasks",
            )
        if m == 0:
            _fail(
                "scenario requires at least one resource",
                code="value_error",
                path="$.resources",
            )
        for index, task in enumerate(self.tasks):
            if isinstance(task.id, bool) or not isinstance(task.id, Integral):
                _fail(
                    "task id must be an integer",
                    code="type_error",
                    path=f"$.tasks[{index}].id",
                )
            if isinstance(task.workload, bool) or not isinstance(task.workload, Real):
                _fail(
                    "task workload must be a number",
                    code="type_error",
                    path=f"$.tasks[{index}].workload",
                )
        for index, resource in enumerate(self.resources):
            if isinstance(resource.id, bool) or not isinstance(resource.id, Integral):
                _fail(
                    "resource id must be an integer",
                    code="type_error",
                    path=f"$.resources[{index}].id",
                )
            if not isinstance(resource.name, str) or not resource.name:
                _fail(
                    "resource name must be a non-empty string",
                    code="value_error",
                    path=f"$.resources[{index}].name",
                )
            if not isinstance(resource.kind, str):
                _fail(
                    "resource kind must be a string",
                    code="type_error",
                    path=f"$.resources[{index}].kind",
                )
            if isinstance(resource.speed, bool) or not isinstance(resource.speed, Real):
                _fail(
                    "resource speed must be a number",
                    code="type_error",
                    path=f"$.resources[{index}].speed",
                )
        if [task.id for task in self.tasks] != list(range(n)):
            _fail(
                "task ids must be contiguous and start at zero",
                code="id_sequence",
                path="$.tasks",
            )
        if [resource.id for resource in self.resources] != list(range(m)):
            _fail(
                "resource ids must be contiguous and start at zero",
                code="id_sequence",
                path="$.resources",
            )
        for index, task in enumerate(self.tasks):
            if not isfinite(float(task.workload)):
                _fail(
                    "task workload must be finite",
                    code="non_finite",
                    path=f"$.tasks[{index}].workload",
                )
            if task.workload <= 0:
                _fail(
                    "task workload must be positive",
                    code="value_error",
                    path=f"$.tasks[{index}].workload",
                )
        for index, resource in enumerate(self.resources):
            if resource.kind not in {"device", "edge", "cloud"}:
                _fail(
                    "resource kind must be device, edge, or cloud",
                    code="value_error",
                    path=f"$.resources[{index}].kind",
                )
            if not isfinite(float(resource.speed)):
                _fail(
                    "resource speed must be finite",
                    code="non_finite",
                    path=f"$.resources[{index}].speed",
                )
            if resource.speed <= 0:
                _fail(
                    "resource speed must be positive",
                    code="value_error",
                    path=f"$.resources[{index}].speed",
                )
        if len(self.bandwidth) != m or any(len(row) != m for row in self.bandwidth):
            _fail(
                "bandwidth must be a resource_count square matrix",
                code="matrix_shape",
                path="$.bandwidth",
            )
        if len(self.latency) != m or any(len(row) != m for row in self.latency):
            _fail(
                "latency must be a resource_count square matrix",
                code="matrix_shape",
                path="$.latency",
            )
        for i in range(m):
            for j in range(m):
                if isinstance(self.bandwidth[i][j], bool) or not isinstance(
                    self.bandwidth[i][j], Real
                ):
                    _fail(
                        "bandwidth entry must be a number",
                        code="type_error",
                        path=f"$.bandwidth[{i}][{j}]",
                    )
                if not isfinite(float(self.bandwidth[i][j])):
                    _fail(
                        "bandwidth entry must be finite",
                        code="non_finite",
                        path=f"$.bandwidth[{i}][{j}]",
                    )
                if self.bandwidth[i][j] <= 0:
                    _fail(
                        "bandwidth entry must be positive",
                        code="value_error",
                        path=f"$.bandwidth[{i}][{j}]",
                    )
                if isinstance(self.latency[i][j], bool) or not isinstance(
                    self.latency[i][j], Real
                ):
                    _fail(
                        "latency entry must be a number",
                        code="type_error",
                        path=f"$.latency[{i}][{j}]",
                    )
                if not isfinite(float(self.latency[i][j])):
                    _fail(
                        "latency entry must be finite",
                        code="non_finite",
                        path=f"$.latency[{i}][{j}]",
                    )
                if self.latency[i][j] < 0:
                    _fail(
                        "latency entry cannot be negative",
                        code="value_error",
                        path=f"$.latency[{i}][{j}]",
                    )
        seen: set[tuple[int, int]] = set()
        indegree = [0] * n
        succ: list[list[int]] = [[] for _ in range(n)]
        for index, edge in enumerate(self.edges):
            if isinstance(edge.source, bool) or not isinstance(edge.source, Integral):
                _fail(
                    "edge source must be an integer",
                    code="type_error",
                    path=f"$.edges[{index}].source",
                )
            if isinstance(edge.target, bool) or not isinstance(edge.target, Integral):
                _fail(
                    "edge target must be an integer",
                    code="type_error",
                    path=f"$.edges[{index}].target",
                )
            if isinstance(edge.data, bool) or not isinstance(edge.data, Real):
                _fail(
                    "edge data must be a number",
                    code="type_error",
                    path=f"$.edges[{index}].data",
                )
            if not 0 <= edge.source < n:
                _fail(
                    "edge source outside task range",
                    code="value_error",
                    path=f"$.edges[{index}].source",
                )
            if not 0 <= edge.target < n:
                _fail(
                    "edge target outside task range",
                    code="value_error",
                    path=f"$.edges[{index}].target",
                )
            if edge.source == edge.target:
                _fail(
                    "dependency edge cannot be a self-loop",
                    code="value_error",
                    path=f"$.edges[{index}]",
                )
            if not isfinite(float(edge.data)):
                _fail(
                    "edge data must be finite",
                    code="non_finite",
                    path=f"$.edges[{index}].data",
                )
            if edge.data < 0:
                _fail(
                    "edge data cannot be negative",
                    code="value_error",
                    path=f"$.edges[{index}].data",
                )
            key = (edge.source, edge.target)
            if key in seen:
                _fail(
                    "duplicate dependency edge",
                    code="duplicate_edge",
                    path=f"$.edges[{index}]",
                )
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
            _fail(
                "task graph must be acyclic",
                code="cycle",
                path="$.edges",
            )

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
        root = _mapping(data, "$")
        _keys(
            root,
            required={"id", "tasks", "resources", "edges", "bandwidth", "latency"},
            optional={"seed"},
            path="$",
        )

        tasks = []
        for index, value in enumerate(_sequence(root["tasks"], "$.tasks")):
            path = f"$.tasks[{index}]"
            item = _mapping(value, path)
            _keys(item, required={"id", "workload"}, path=path)
            tasks.append(
                Task(
                    id=_integer(item["id"], f"{path}.id"),
                    workload=_number(item["workload"], f"{path}.workload"),
                )
            )

        resources = []
        for index, value in enumerate(_sequence(root["resources"], "$.resources")):
            path = f"$.resources[{index}]"
            item = _mapping(value, path)
            _keys(item, required={"id", "name", "kind", "speed"}, path=path)
            resources.append(
                Resource(
                    id=_integer(item["id"], f"{path}.id"),
                    name=_string(item["name"], f"{path}.name"),
                    kind=_string(item["kind"], f"{path}.kind"),
                    speed=_number(item["speed"], f"{path}.speed"),
                )
            )

        edges = []
        for index, value in enumerate(_sequence(root["edges"], "$.edges")):
            path = f"$.edges[{index}]"
            item = _mapping(value, path)
            _keys(item, required={"source", "target", "data"}, path=path)
            edges.append(
                Edge(
                    source=_integer(item["source"], f"{path}.source"),
                    target=_integer(item["target"], f"{path}.target"),
                    data=_number(item["data"], f"{path}.data"),
                )
            )

        return cls(
            id=_string(root["id"], "$.id"),
            seed=_integer(root.get("seed", 0), "$.seed"),
            tasks=tuple(tasks),
            resources=tuple(resources),
            edges=tuple(edges),
            bandwidth=_matrix(root["bandwidth"], "$.bandwidth"),
            latency=_matrix(root["latency"], "$.latency"),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "Scenario":
        source = Path(path)

        def reject_constant(value: str) -> None:
            _fail(
                f"JSON constant {value} is not allowed",
                code="non_finite",
                path="$",
            )

        try:
            text = source.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as error:
            _fail(
                f"scenario file must be UTF-8: {error.reason}",
                code="encoding_error",
                path="$",
            )
        try:
            payload = json.loads(
                text,
                parse_constant=reject_constant,
            )
        except json.JSONDecodeError as error:
            message = (
                f"invalid JSON at line {error.lineno}, "
                f"column {error.colno}: {error.msg}"
            )
            _fail(
                message,
                code="json_syntax",
                path="$",
            )
        return cls.from_dict(payload)

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
