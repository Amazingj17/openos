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


def _string_tuple(value: Any, path: str) -> tuple[str, ...]:
    values = _sequence(value, path)
    result = tuple(_string(item, f"{path}[{index}]") for index, item in enumerate(values))
    if len(set(result)) != len(result):
        _fail("array entries must be unique", code="duplicate_value", path=path)
    return result


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
    cpu_cores: int = 1
    memory_capacity: float = 1.0
    has_accelerator: bool = False
    features: tuple[str, ...] = ()


@dataclass(frozen=True)
class Task:
    id: int
    workload: float
    cpu_cores_required: int = 1
    memory_required: float = 0.0
    accelerator_required: bool = False
    required_features: tuple[str, ...] = ()
    task_type: str = "generic"


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
    execution_times: tuple[tuple[float, ...], ...] | None = None

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
        if self.execution_times is not None:
            return self.execution_times[task_id][resource_id]
        return self.tasks[task_id].workload / self.resources[resource_id].speed

    def resource_is_compatible(self, task_id: int, resource_id: int) -> bool:
        task = self.tasks[task_id]
        resource = self.resources[resource_id]
        return (
            task.cpu_cores_required <= resource.cpu_cores
            and task.memory_required <= resource.memory_capacity
            and (not task.accelerator_required or resource.has_accelerator)
            and set(task.required_features).issubset(resource.features)
        )

    def compatible_resources(self, task_id: int) -> tuple[int, ...]:
        return tuple(
            resource.id
            for resource in self.resources
            if self.resource_is_compatible(task_id, resource.id)
        )

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
            if isinstance(task.memory_required, bool) or not isinstance(
                task.memory_required, Real
            ):
                _fail(
                    "task memory requirement must be a number",
                    code="type_error",
                    path=f"$.tasks[{index}].memory_required",
                )
            if (
                isinstance(task.cpu_cores_required, bool)
                or not isinstance(task.cpu_cores_required, Integral)
                or task.cpu_cores_required <= 0
            ):
                _fail(
                    "task CPU core requirement must be a positive integer",
                    code="value_error",
                    path=f"$.tasks[{index}].cpu_cores_required",
                )
            if not isinstance(task.accelerator_required, bool):
                _fail(
                    "task accelerator requirement must be a boolean",
                    code="type_error",
                    path=f"$.tasks[{index}].accelerator_required",
                )
            if not isinstance(task.task_type, str) or not task.task_type:
                _fail(
                    "task type must be a non-empty string",
                    code="value_error",
                    path=f"$.tasks[{index}].task_type",
                )
            if any(not isinstance(item, str) or not item for item in task.required_features):
                _fail(
                    "task required features must be non-empty strings",
                    code="type_error",
                    path=f"$.tasks[{index}].required_features",
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
            if isinstance(resource.memory_capacity, bool) or not isinstance(
                resource.memory_capacity, Real
            ):
                _fail(
                    "resource memory capacity must be a number",
                    code="type_error",
                    path=f"$.resources[{index}].memory_capacity",
                )
            if (
                isinstance(resource.cpu_cores, bool)
                or not isinstance(resource.cpu_cores, Integral)
                or resource.cpu_cores <= 0
            ):
                _fail(
                    "resource CPU cores must be a positive integer",
                    code="value_error",
                    path=f"$.resources[{index}].cpu_cores",
                )
            if not isinstance(resource.has_accelerator, bool):
                _fail(
                    "resource accelerator flag must be a boolean",
                    code="type_error",
                    path=f"$.resources[{index}].has_accelerator",
                )
            if any(not isinstance(item, str) or not item for item in resource.features):
                _fail(
                    "resource features must be non-empty strings",
                    code="type_error",
                    path=f"$.resources[{index}].features",
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
            if not isfinite(float(task.memory_required)) or task.memory_required < 0:
                _fail(
                    "task memory requirement must be finite and non-negative",
                    code="value_error",
                    path=f"$.tasks[{index}].memory_required",
                )
            if len(set(task.required_features)) != len(task.required_features):
                _fail(
                    "task required features must be unique",
                    code="duplicate_value",
                    path=f"$.tasks[{index}].required_features",
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
            if (
                not isfinite(float(resource.memory_capacity))
                or resource.memory_capacity <= 0
            ):
                _fail(
                    "resource memory capacity must be finite and positive",
                    code="value_error",
                    path=f"$.resources[{index}].memory_capacity",
                )
            if len(set(resource.features)) != len(resource.features):
                _fail(
                    "resource features must be unique",
                    code="duplicate_value",
                    path=f"$.resources[{index}].features",
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
        if self.execution_times is not None:
            if len(self.execution_times) != n or any(
                len(row) != m for row in self.execution_times
            ):
                _fail(
                    "execution_times must be a task_count by resource_count matrix",
                    code="matrix_shape",
                    path="$.execution_times",
                )
            for task_id, row in enumerate(self.execution_times):
                for resource_id, value in enumerate(row):
                    if isinstance(value, bool) or not isinstance(value, Real):
                        _fail(
                            "execution time must be a number",
                            code="type_error",
                            path=f"$.execution_times[{task_id}][{resource_id}]",
                        )
                    if not isfinite(float(value)) or value <= 0:
                        _fail(
                            "execution time must be finite and positive",
                            code="value_error",
                            path=f"$.execution_times[{task_id}][{resource_id}]",
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
        for task in self.tasks:
            if not self.compatible_resources(task.id):
                _fail(
                    "task has no compatible cloud-edge-device resource",
                    code="unschedulable_task",
                    path=f"$.tasks[{task.id}]",
                )

    def to_dict(self) -> dict[str, Any]:
        tasks: list[dict[str, Any]] = []
        for task in self.tasks:
            item: dict[str, Any] = {"id": task.id, "workload": task.workload}
            if task.cpu_cores_required != 1:
                item["cpu_cores_required"] = task.cpu_cores_required
            if task.memory_required != 0.0:
                item["memory_required"] = task.memory_required
            if task.accelerator_required:
                item["accelerator_required"] = True
            if task.required_features:
                item["required_features"] = list(task.required_features)
            if task.task_type != "generic":
                item["task_type"] = task.task_type
            tasks.append(item)
        resources: list[dict[str, Any]] = []
        for resource in self.resources:
            item = {
                "id": resource.id,
                "name": resource.name,
                "kind": resource.kind,
                "speed": resource.speed,
            }
            if resource.memory_capacity != 1.0:
                item["memory_capacity"] = resource.memory_capacity
            if resource.cpu_cores != 1:
                item["cpu_cores"] = resource.cpu_cores
            if resource.has_accelerator:
                item["has_accelerator"] = True
            if resource.features:
                item["features"] = list(resource.features)
            resources.append(item)
        payload: dict[str, Any] = {
            "id": self.id,
            "seed": self.seed,
            "tasks": tasks,
            "resources": resources,
            "edges": [asdict(edge) for edge in self.edges],
            "bandwidth": [list(row) for row in self.bandwidth],
            "latency": [list(row) for row in self.latency],
        }
        if self.execution_times is not None:
            payload["execution_times"] = [list(row) for row in self.execution_times]
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scenario":
        root = _mapping(data, "$")
        _keys(
            root,
            required={"id", "tasks", "resources", "edges", "bandwidth", "latency"},
            optional={"seed", "execution_times"},
            path="$",
        )

        tasks = []
        for index, value in enumerate(_sequence(root["tasks"], "$.tasks")):
            path = f"$.tasks[{index}]"
            item = _mapping(value, path)
            _keys(
                item,
                required={"id", "workload"},
                optional={
                    "memory_required",
                    "cpu_cores_required",
                    "accelerator_required",
                    "required_features",
                    "task_type",
                },
                path=path,
            )
            tasks.append(
                Task(
                    id=_integer(item["id"], f"{path}.id"),
                    workload=_number(item["workload"], f"{path}.workload"),
                    cpu_cores_required=_integer(
                        item.get("cpu_cores_required", 1),
                        f"{path}.cpu_cores_required",
                    ),
                    memory_required=_number(
                        item.get("memory_required", 0.0),
                        f"{path}.memory_required",
                    ),
                    accelerator_required=item.get("accelerator_required", False),
                    required_features=_string_tuple(
                        item.get("required_features", []),
                        f"{path}.required_features",
                    ),
                    task_type=_string(item.get("task_type", "generic"), f"{path}.task_type"),
                )
            )

        resources = []
        for index, value in enumerate(_sequence(root["resources"], "$.resources")):
            path = f"$.resources[{index}]"
            item = _mapping(value, path)
            _keys(
                item,
                required={"id", "name", "kind", "speed"},
                optional={
                    "cpu_cores",
                    "memory_capacity",
                    "has_accelerator",
                    "features",
                },
                path=path,
            )
            resources.append(
                Resource(
                    id=_integer(item["id"], f"{path}.id"),
                    name=_string(item["name"], f"{path}.name"),
                    kind=_string(item["kind"], f"{path}.kind"),
                    speed=_number(item["speed"], f"{path}.speed"),
                    cpu_cores=_integer(
                        item.get("cpu_cores", 1), f"{path}.cpu_cores"
                    ),
                    memory_capacity=_number(
                        item.get("memory_capacity", 1.0),
                        f"{path}.memory_capacity",
                    ),
                    has_accelerator=item.get("has_accelerator", False),
                    features=_string_tuple(
                        item.get("features", []), f"{path}.features"
                    ),
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
            execution_times=(
                _matrix(root["execution_times"], "$.execution_times")
                if "execution_times" in root
                else None
            ),
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


def generate_complex_scenario(
    seed: int,
    task_count: int = 100,
    resource_count: int = 9,
    edge_probability: float = 0.08,
    scenario_id: str | None = None,
) -> Scenario:
    """Generate a constraint- and data-aware cloud-edge-device scenario.

    The generator deliberately keeps the competition's three resource kinds while
    allowing multiple heterogeneous instances of each kind.  Its explicit
    task-by-resource execution matrix breaks the overly simple assumption that
    one machine is uniformly faster for every task type.
    """

    if task_count < 2:
        raise ValueError("task_count must be at least two")
    if resource_count < 3:
        raise ValueError("complex scenarios require at least three resources")
    if not 0 <= edge_probability <= 1:
        raise ValueError("edge_probability must be between zero and one")
    rng = np.random.default_rng(seed)
    kind_counts = {"device": 0, "edge": 0, "cloud": 0}
    speed_base = {"device": 1.0, "edge": 4.0, "cloud": 9.0}
    memory_range = {
        "device": (2.0, 8.0),
        "edge": (16.0, 64.0),
        "cloud": (64.0, 256.0),
    }
    core_range = {"device": (2, 8), "edge": (8, 32), "cloud": (32, 96)}
    resources: list[Resource] = []
    for resource_id in range(resource_count):
        kind = _resource_kind(resource_id, resource_count)
        suffix = kind_counts[kind]
        kind_counts[kind] += 1
        has_accelerator = kind == "cloud" or (
            kind == "edge" and suffix % 2 == 0
        ) or (kind == "device" and suffix % 3 == 0)
        features = {"low_latency"} if kind != "cloud" else {"high_bandwidth"}
        if kind == "edge":
            features.add("high_bandwidth")
        if kind == "device":
            features.add("trusted_local")
        if has_accelerator:
            features.add("accelerator")
        resources.append(
            Resource(
                id=resource_id,
                name=f"{kind}-{suffix}",
                kind=kind,
                speed=float(speed_base[kind] * rng.uniform(0.75, 1.25)),
                cpu_cores=int(rng.integers(core_range[kind][0], core_range[kind][1] + 1)),
                memory_capacity=float(rng.uniform(*memory_range[kind])),
                has_accelerator=has_accelerator,
                features=tuple(sorted(features)),
            )
        )

    task_types = ("cpu", "data", "gpu")
    tasks: list[Task] = []
    for task_id in range(task_count):
        task_type = str(rng.choice(task_types, p=(0.5, 0.3, 0.2)))
        accelerator_required = task_type == "gpu"
        required_features: tuple[str, ...] = ()
        if task_type == "data" and rng.random() < 0.35:
            required_features = ("high_bandwidth",)
        memory_required = float(
            rng.uniform(0.25, 6.0 if task_type != "gpu" else 12.0)
        )
        core_limit = 4 if task_type != "gpu" else 8
        tasks.append(
            Task(
                id=task_id,
                workload=float(rng.uniform(5.0, 100.0)),
                cpu_cores_required=int(rng.integers(1, core_limit + 1)),
                memory_required=memory_required,
                accelerator_required=accelerator_required,
                required_features=required_features,
                task_type=task_type,
            )
        )

    edge_pairs: set[tuple[int, int]] = set()
    for target in range(1, task_count):
        lower = max(0, target - max(4, task_count // 8))
        edge_pairs.add((int(rng.integers(lower, target)), target))
    for source in range(task_count):
        for target in range(source + 1, task_count):
            if rng.random() < edge_probability:
                edge_pairs.add((source, target))
    edges = tuple(
        Edge(source, target, float(rng.lognormal(mean=1.2, sigma=0.9)))
        for source, target in sorted(edge_pairs)
    )

    bandwidth: list[list[float]] = []
    latency: list[list[float]] = []
    for source in resources:
        bw_row: list[float] = []
        latency_row: list[float] = []
        for target in resources:
            if source.id == target.id:
                bw_row.append(1e9)
                latency_row.append(0.0)
            else:
                base_bw, base_latency = _link_profile(source.kind, target.kind)
                bw_row.append(float(base_bw * rng.uniform(0.65, 1.35)))
                latency_row.append(float(base_latency * rng.uniform(0.7, 1.4)))
        bandwidth.append(bw_row)
        latency.append(latency_row)

    affinity = {
        "cpu": {"device": 0.9, "edge": 1.0, "cloud": 1.2},
        "data": {"device": 1.25, "edge": 1.4, "cloud": 0.8},
        "gpu": {"device": 2.0, "edge": 4.0, "cloud": 7.0},
    }
    execution_times = tuple(
        tuple(
            float(
                task.workload
                / max(resource.speed * affinity[task.task_type][resource.kind], 1e-9)
            )
            for resource in resources
        )
        for task in tasks
    )
    return Scenario(
        id=scenario_id or f"complex-{seed}",
        seed=seed,
        tasks=tuple(tasks),
        resources=tuple(resources),
        edges=edges,
        bandwidth=tuple(tuple(row) for row in bandwidth),
        latency=tuple(tuple(row) for row in latency),
        execution_times=execution_times,
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
