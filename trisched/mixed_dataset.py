from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .data_sources import (
    ResourceTopology,
    load_dagbench_json,
    load_stg_json_v2,
    load_topology_zoo_graph,
    load_topology_zoo_graphml,
    topology_from_scenario,
)
from .scenario import Edge, Resource, Scenario, Task, generate_complex_scenario


ADAPTER_VERSION = "mixed-cloud-edge-device-v1"
SPLIT_NAMES = (
    "train",
    "id_validation",
    "dag_ood",
    "network_ood",
    "joint_ood",
)


@dataclass(frozen=True)
class TaskSample:
    key: str
    family: str
    source: str
    path: str | None
    scenario: Scenario
    raw_sha256: str | None


@dataclass(frozen=True)
class NetworkSample:
    key: str
    family: str
    source: str
    path: str | None
    topology: ResourceTopology
    raw_sha256: str | None


@dataclass(frozen=True)
class EnhancedTaskGraph:
    id: str
    source: str
    family: str
    seed: int
    tasks: tuple[Task, ...]
    edges: tuple[Edge, ...]


@dataclass(frozen=True)
class EnhancedResourceGraph:
    id: str
    source: str
    family: str
    topology: ResourceTopology


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    content = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _filename(key: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", key).strip("._-") or "item"
    suffix = hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]
    return f"{readable}-{suffix}.json"


def _task_payload(item: TaskSample) -> dict[str, Any]:
    return {
        "format_version": 1,
        "kind": "enhanced_task_graph",
        "adapter_version": ADAPTER_VERSION,
        "id": item.key,
        "source": item.source,
        "family": item.family,
        "seed": item.scenario.seed,
        "units": {"workload": "source_cost_unit", "memory_required": "MiB", "edge_data": "MB"},
        "origin": {"path": item.path, "raw_sha256": item.raw_sha256},
        "tasks": [
            {
                "id": task.id,
                "workload": task.workload,
                "cpu_cores_required": task.cpu_cores_required,
                "memory_required": task.memory_required,
                "accelerator_required": task.accelerator_required,
                "required_features": list(task.required_features),
                "task_type": task.task_type,
            }
            for task in item.scenario.tasks
        ],
        "edges": [
            {"source": edge.source, "target": edge.target, "data": edge.data}
            for edge in item.scenario.edges
        ],
    }


def _resource_payload(item: NetworkSample) -> dict[str, Any]:
    topology = item.topology
    return {
        "format_version": 1,
        "kind": "enhanced_resource_graph",
        "adapter_version": ADAPTER_VERSION,
        "id": item.key,
        "source": item.source,
        "family": item.family,
        "units": {"memory_capacity": "MiB", "bandwidth": "MB/s", "latency": "seconds"},
        "origin": {"path": item.path, "raw_sha256": item.raw_sha256},
        "resources": [
            {
                "id": resource.id,
                "name": resource.name,
                "kind": resource.kind,
                "speed": resource.speed,
                "cpu_cores": resource.cpu_cores,
                "memory_capacity": resource.memory_capacity,
                "has_accelerator": resource.has_accelerator,
                "features": list(resource.features),
            }
            for resource in topology.resources
        ],
        "bandwidth": [list(row) for row in topology.bandwidth],
        "latency": [list(row) for row in topology.latency],
    }


def load_enhanced_task_graph(path: str | Path) -> EnhancedTaskGraph:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if payload.get("kind") != "enhanced_task_graph" or payload.get("format_version") != 1:
        raise ValueError("unsupported enhanced task graph format")
    tasks = tuple(
        Task(
            id=int(item["id"]),
            workload=float(item["workload"]),
            cpu_cores_required=int(item["cpu_cores_required"]),
            memory_required=float(item["memory_required"]),
            accelerator_required=bool(item["accelerator_required"]),
            required_features=tuple(str(value) for value in item["required_features"]),
            task_type=str(item["task_type"]),
        )
        for item in payload["tasks"]
    )
    edges = tuple(
        Edge(int(item["source"]), int(item["target"]), float(item["data"]))
        for item in payload["edges"]
    )
    return EnhancedTaskGraph(
        str(payload["id"]),
        str(payload["source"]),
        str(payload["family"]),
        int(payload["seed"]),
        tasks,
        edges,
    )


def load_enhanced_resource_graph(path: str | Path) -> EnhancedResourceGraph:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    if payload.get("kind") != "enhanced_resource_graph" or payload.get("format_version") != 1:
        raise ValueError("unsupported enhanced resource graph format")
    resources = tuple(
        Resource(
            id=int(item["id"]),
            name=str(item["name"]),
            kind=str(item["kind"]),
            speed=float(item["speed"]),
            cpu_cores=int(item["cpu_cores"]),
            memory_capacity=float(item["memory_capacity"]),
            has_accelerator=bool(item["has_accelerator"]),
            features=tuple(str(value) for value in item["features"]),
        )
        for item in payload["resources"]
    )
    topology = ResourceTopology(
        resources,
        tuple(tuple(float(value) for value in row) for row in payload["bandwidth"]),
        tuple(tuple(float(value) for value in row) for row in payload["latency"]),
        source.as_posix(),
    )
    return EnhancedResourceGraph(
        str(payload["id"]), str(payload["source"]), str(payload["family"]), topology
    )


def _task_graph_as_sample(
    graph: EnhancedTaskGraph,
    *,
    path: str,
    raw_sha256: str | None,
) -> TaskSample:
    maximum_cores = max(task.cpu_cores_required for task in graph.tasks)
    maximum_memory = max(task.memory_required for task in graph.tasks)
    features = set().union(*(set(task.required_features) for task in graph.tasks))
    resource = Resource(
        0,
        "dataset-validation-cloud",
        "cloud",
        1.0,
        maximum_cores,
        maximum_memory,
        True,
        tuple(sorted(features | {"cpu", "gpu", "accelerator"})),
    )
    scenario = Scenario(
        id=graph.id,
        seed=graph.seed,
        tasks=graph.tasks,
        resources=(resource,),
        edges=graph.edges,
        bandwidth=((1e9,),),
        latency=((0.0,),),
    )
    return TaskSample(
        graph.id, graph.family, graph.source, path, scenario, raw_sha256
    )


def _load_separated_catalog(
    root_manifest_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8-sig"))
    if root_manifest.get("kind") != "trisched_separated_enhanced_dataset":
        raise ValueError("invalid separated enhanced dataset root manifest")
    root = root_manifest_path.parent
    task_manifest = json.loads(
        (root / root_manifest["task_manifest"]).read_text(encoding="utf-8-sig")
    )
    resource_manifest = json.loads(
        (root / root_manifest["resource_manifest"]).read_text(encoding="utf-8-sig")
    )
    tasks: list[TaskSample] = []
    for entry in task_manifest["entries"]:
        path = root / entry["file"]
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if _payload_sha256(payload) != str(entry["enhanced_sha256"]):
            raise ValueError(f"enhanced task graph hash mismatch: {entry['file']}")
        graph = load_enhanced_task_graph(path)
        tasks.append(
            _task_graph_as_sample(
                graph,
                path=_portable_path(path, project_root, str(entry["enhanced_sha256"])),
                raw_sha256=entry.get("raw_sha256"),
            )
        )
    networks: list[NetworkSample] = []
    for entry in resource_manifest["entries"]:
        path = root / entry["file"]
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if _payload_sha256(payload) != str(entry["enhanced_sha256"]):
            raise ValueError(f"enhanced resource graph hash mismatch: {entry['file']}")
        graph = load_enhanced_resource_graph(path)
        networks.append(
            NetworkSample(
                graph.id,
                graph.family,
                graph.source,
                _portable_path(path, project_root, str(entry["enhanced_sha256"])),
                graph.topology,
                entry.get("raw_sha256"),
            )
        )
    task_sources = {name: sum(item.source == name for item in tasks) for name in {item.source for item in tasks}}
    network_sources = {name: sum(item.source == name for item in networks) for name in {item.source for item in networks}}
    return {
        "tasks": tasks,
        "networks": networks,
        "discovered": {
            "stg": task_sources.get("stg", 0),
            "dagbench_tasks": task_sources.get("dagbench", 0),
            "dagbench_networks": network_sources.get("dagbench", 0),
            "topology_zoo": network_sources.get("topology_zoo", 0),
            "synthetic_tasks": task_sources.get("synthetic", 0),
            "synthetic_networks": network_sources.get("synthetic", 0),
        },
    }


def _stable_int(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _project_root(config_dir: Path) -> Path:
    """Find the checkout root without baking the current machine into output."""

    candidate = config_dir.resolve()
    for directory in (candidate, *candidate.parents):
        if (directory / "pyproject.toml").is_file():
            return directory
    return candidate


def _portable_path(path: Path, root: Path, raw_sha256: str) -> str:
    """Return a manifest locator that never exposes a host absolute path."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        # Inputs deliberately kept outside a checkout cannot have a reusable
        # relative filesystem locator.  The content-addressed URI remains safe
        # and unambiguous; the raw SHA-256 is also stored beside it.
        return f"external-sha256://{raw_sha256}/{path.name}"


def _resolve_pattern(config_dir: Path, pattern: str) -> tuple[Path, str]:
    normalized = pattern.replace("\\", "/")
    wildcard = min(
        (normalized.find(char) for char in "*?[" if char in normalized),
        default=len(normalized),
    )
    prefix = normalized[:wildcard]
    root_text = prefix.rsplit("/", 1)[0] if "/" in prefix else "."
    root = Path(root_text)
    if not root.is_absolute():
        # Config paths are repository-relative by convention; falling back to
        # the config directory makes temporary/test configurations portable.
        repository_candidate = Path.cwd() / root
        root = repository_candidate if repository_candidate.exists() else config_dir / root
    relative_pattern = normalized[len(root_text) :].lstrip("/")
    return root.resolve(), relative_pattern or "*"


def _glob(config_dir: Path, patterns: Sequence[str]) -> list[Path]:
    files: set[Path] = set()
    for pattern in patterns:
        root, relative = _resolve_pattern(config_dir, pattern)
        if root.exists():
            files.update(path.resolve() for path in root.glob(relative) if path.is_file())
    return sorted(files, key=lambda path: path.as_posix())


def _dagbench_family(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    name = str(payload.get("name") or path.parent.name)
    return name.split(".", 1)[0]


def _load_stg_split_map(manifest_path: Path | None, raw_root: Path) -> dict[Path, str]:
    if manifest_path is None or not manifest_path.is_file():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    return {
        (raw_root / str(entry["source"])).resolve(): str(entry["split"])
        for entry in payload.get("entries", [])
    }


def _synthetic_task(seed: int, index: int, task_range: tuple[int, int]) -> TaskSample:
    rng = np.random.default_rng(_stable_int(seed, "synthetic-task", index))
    count = int(rng.integers(task_range[0], task_range[1] + 1))
    scenario = generate_complex_scenario(
        int(_stable_int(seed, "synthetic-task-scenario", index) % (2**31)),
        task_count=count,
        resource_count=6,
        edge_probability=0.06,
        scenario_id=f"synthetic-task-{index:04d}",
    )
    # The historical complex generator expresses memory in GiB.  Mixed data
    # uses MiB so it can be combined with STG without an implicit unit change.
    tasks = tuple(replace(task, memory_required=task.memory_required * 1024.0) for task in scenario.tasks)
    resources = tuple(
        replace(resource, memory_capacity=resource.memory_capacity * 1024.0)
        for resource in scenario.resources
    )
    scenario = replace(scenario, tasks=tasks, resources=resources)
    return TaskSample(scenario.id, "synthetic", "synthetic", None, scenario, None)


def _synthetic_network(seed: int, index: int, resource_count: int) -> NetworkSample:
    scenario = generate_complex_scenario(
        int(_stable_int(seed, "synthetic-network", index) % (2**31)),
        task_count=2,
        resource_count=resource_count,
        edge_probability=0.0,
        scenario_id=f"synthetic-network-{index:04d}",
    )
    resources = tuple(
        replace(resource, memory_capacity=resource.memory_capacity * 1024.0)
        for resource in scenario.resources
    )
    topology = ResourceTopology(
        resources,
        scenario.bandwidth,
        scenario.latency,
        f"synthetic://network/{index}",
    )
    return NetworkSample(
        scenario.id, "synthetic", "synthetic", None, topology, None
    )


def discover_catalog(
    config: Mapping[str, Any],
    config_dir: Path,
    *,
    prefer_exported: bool = True,
) -> dict[str, Any]:
    seed = int(config["seed"])
    dataset = config["dataset"]
    sources = dataset.get("sources", {})
    task_range = tuple(int(value) for value in dataset.get("task_range", (24, 40)))
    resource_count = int(dataset.get("resource_count", 6))
    project_root = _project_root(config_dir)
    enhanced = sources.get("enhanced", {})
    enhanced_manifest = enhanced.get("manifest")
    if prefer_exported and bool(enhanced.get("prefer", False)) and enhanced_manifest:
        candidate = Path(str(enhanced_manifest))
        if not candidate.is_absolute():
            candidate = project_root / candidate
        if candidate.is_file():
            return _load_separated_catalog(candidate.resolve(), project_root)

    dagbench_files = _glob(config_dir, sources.get("dagbench", {}).get("patterns", []))
    stg_files = _glob(config_dir, sources.get("stg", {}).get("patterns", []))
    zoo_files = _glob(config_dir, sources.get("topology_zoo", {}).get("patterns", []))
    stg_raw_root = Path(sources.get("stg", {}).get("raw_root", "."))
    if not stg_raw_root.is_absolute():
        stg_raw_root = (Path.cwd() / stg_raw_root).resolve()
    manifest_value = sources.get("stg", {}).get("manifest")
    manifest_path = (Path.cwd() / manifest_value).resolve() if manifest_value else None
    stg_split_map = _load_stg_split_map(manifest_path, stg_raw_root)

    tasks: list[TaskSample] = []
    networks: list[NetworkSample] = []
    for path in stg_files:
        scenario = load_stg_json_v2(path)
        split = stg_split_map.get(path, "unassigned")
        raw_hash = _sha256(path)
        tasks.append(
            TaskSample(
                f"stg:{path.stem}",
                f"stg-{split}",
                "stg",
                _portable_path(path, project_root, raw_hash),
                scenario,
                raw_hash,
            )
        )
    for index, path in enumerate(dagbench_files):
        scenario = load_dagbench_json(
            path,
            seed=int(_stable_int(seed, "dagbench", path.as_posix()) % (2**31)),
        )
        family = _dagbench_family(path)
        raw_hash = _sha256(path)
        tasks.append(
            TaskSample(
                f"dagbench:{scenario.id}",
                family,
                "dagbench",
                _portable_path(path, project_root, raw_hash),
                scenario,
                raw_hash,
            )
        )
        networks.append(
            NetworkSample(
                f"dagbench-network:{scenario.id}",
                family,
                "dagbench",
                _portable_path(path, project_root, raw_hash),
                topology_from_scenario(scenario, source=str(path)),
                raw_hash,
            )
        )
    for path in zoo_files:
        topology = (
            load_topology_zoo_graphml(path)
            if path.suffix.lower() == ".graphml"
            else load_topology_zoo_graph(path)
        )
        raw_hash = _sha256(path)
        networks.append(
            NetworkSample(
                f"topology-zoo:{path.stem}",
                path.stem.lower(),
                "topology_zoo",
                _portable_path(path, project_root, raw_hash),
                topology,
                raw_hash,
            )
        )

    synthetic_count = max(int(dataset.get("synthetic_catalog_size", 32)), 1)
    tasks.extend(_synthetic_task(seed, index, task_range) for index in range(synthetic_count))
    networks.extend(
        _synthetic_network(seed, index, resource_count) for index in range(synthetic_count)
    )
    return {
        "tasks": tasks,
        "networks": networks,
        "discovered": {
            "stg": len(stg_files),
            "dagbench_tasks": len(dagbench_files),
            "dagbench_networks": len(dagbench_files),
            "topology_zoo": len(zoo_files),
            "synthetic_tasks": synthetic_count,
            "synthetic_networks": synthetic_count,
        },
    }


def _weighted_choice(
    rng: np.random.Generator,
    pools: Mapping[str, Sequence[Any]],
    weights: Mapping[str, float],
    *,
    context: str,
) -> Any:
    available = [name for name, pool in pools.items() if pool and float(weights.get(name, 0.0)) > 0]
    if not available:
        raise ValueError(f"no available source for {context}")
    probabilities = np.asarray([float(weights[name]) for name in available], dtype=np.float64)
    probabilities /= np.sum(probabilities)
    source = available[int(rng.choice(len(available), p=probabilities))]
    pool = pools[source]
    return pool[int(rng.integers(0, len(pool)))]


def _source_schedule(
    rng: np.random.Generator,
    pools: Mapping[str, Sequence[Any]],
    weights: Mapping[str, float],
    count: int,
    *,
    context: str,
) -> list[str]:
    """Create a shuffled, weighted quota with coverage for small experiments."""

    available = [name for name, pool in pools.items() if pool and float(weights.get(name, 0.0)) > 0]
    if not available:
        raise ValueError(f"no available source for {context}")
    quotas = {name: 0 for name in available}
    remaining = count
    if count >= len(available):
        for name in available:
            quotas[name] = 1
        remaining -= len(available)
    probabilities = np.asarray([float(weights[name]) for name in available], dtype=np.float64)
    probabilities /= np.sum(probabilities)
    expected = probabilities * remaining
    for name, value in zip(available, np.floor(expected).astype(int)):
        quotas[name] += int(value)
    leftover = count - sum(quotas.values())
    order = sorted(
        range(len(available)),
        key=lambda index: (-(expected[index] - np.floor(expected[index])), available[index]),
    )
    for index in order[:leftover]:
        quotas[available[index]] += 1
    schedule = [name for name in available for _ in range(quotas[name])]
    rng.shuffle(schedule)
    return schedule


def _compatible_topology(tasks: Sequence[Task], topology: ResourceTopology) -> ResourceTopology:
    resources = list(topology.resources)
    if not resources:
        raise ValueError("resource topology is empty")
    maximum_cores = max(task.cpu_cores_required for task in tasks)
    maximum_memory = max(task.memory_required for task in tasks)
    required_features = set().union(*(set(task.required_features) for task in tasks))
    target = max(
        range(len(resources)),
        key=lambda index: (
            resources[index].kind == "cloud",
            resources[index].speed,
            resources[index].cpu_cores,
        ),
    )
    resource = resources[target]
    resources[target] = replace(
        resource,
        cpu_cores=max(resource.cpu_cores, maximum_cores),
        memory_capacity=max(resource.memory_capacity, maximum_memory),
        has_accelerator=True,
        features=tuple(sorted(set(resource.features) | required_features | {"cpu", "gpu", "accelerator"})),
    )
    return ResourceTopology(tuple(resources), topology.bandwidth, topology.latency, topology.source)


def _limit_topology(topology: ResourceTopology, maximum: int | None) -> ResourceTopology:
    if maximum is None or len(topology.resources) <= maximum:
        return topology
    if maximum < 3:
        raise ValueError("max_resource_nodes must be at least three")
    by_kind = {
        kind: [resource.id for resource in topology.resources if resource.kind == kind]
        for kind in ("device", "edge", "cloud")
    }
    selected: list[int] = []
    # Round-robin selection preserves all available tiers and avoids retaining
    # only the high-degree/cloud end of a real topology.
    while len(selected) < maximum:
        changed = False
        for kind in ("device", "edge", "cloud"):
            if by_kind[kind] and len(selected) < maximum:
                selected.append(by_kind[kind].pop(0))
                changed = True
        if not changed:
            break
    selected.sort()
    resources = tuple(
        replace(topology.resources[old_id], id=new_id)
        for new_id, old_id in enumerate(selected)
    )
    bandwidth = tuple(tuple(topology.bandwidth[i][j] for j in selected) for i in selected)
    latency = tuple(tuple(topology.latency[i][j] for j in selected) for i in selected)
    return ResourceTopology(resources, bandwidth, latency, topology.source)


def assemble_scenario(
    task_sample: TaskSample,
    network_sample: NetworkSample,
    *,
    scenario_id: str,
    seed: int,
    max_resource_nodes: int | None = None,
) -> Scenario:
    topology = _limit_topology(network_sample.topology, max_resource_nodes)
    topology = _compatible_topology(task_sample.scenario.tasks, topology)
    affinity = {
        "cpu": {"device": 0.9, "edge": 1.0, "cloud": 1.2},
        "data": {"device": 1.2, "edge": 1.5, "cloud": 0.85},
        "gpu": {"device": 1.5, "edge": 4.0, "cloud": 7.0},
        "generic": {"device": 1.0, "edge": 1.0, "cloud": 1.0},
    }
    execution_times = tuple(
        tuple(
            float(task.workload / max(resource.speed * affinity.get(task.task_type, affinity["generic"])[resource.kind], 1e-9))
            for resource in topology.resources
        )
        for task in task_sample.scenario.tasks
    )
    return Scenario(
        id=scenario_id,
        seed=seed,
        tasks=task_sample.scenario.tasks,
        resources=topology.resources,
        edges=task_sample.scenario.edges,
        bandwidth=topology.bandwidth,
        latency=topology.latency,
        execution_times=execution_times,
    )


def _partition_catalog(config: Mapping[str, Any], catalog: Mapping[str, Any]) -> dict[str, Any]:
    dataset = config["dataset"]
    heldout_dag = set(dataset.get("heldout_dag_families", ["scientific", "ml", "uav", "v2x"]))
    heldout_network = {str(value).lower() for value in dataset.get("heldout_topologies", [])}
    tasks: Sequence[TaskSample] = catalog["tasks"]
    networks: Sequence[NetworkSample] = catalog["networks"]

    task_train: dict[str, list[TaskSample]] = {"stg": [], "dagbench": [], "synthetic": []}
    task_id: dict[str, list[TaskSample]] = {"stg": [], "dagbench": [], "synthetic": []}
    task_ood: list[TaskSample] = []
    for item in tasks:
        if item.source == "dagbench" and item.family in heldout_dag:
            task_ood.append(item)
        elif item.source == "stg" and item.family == "stg-validation":
            task_id["stg"].append(item)
        elif item.source == "stg" and item.family == "stg-test":
            continue
        elif _stable_int("id-partition", item.key) % 5 == 0:
            task_id[item.source].append(item)
        else:
            task_train[item.source].append(item)
    for source in task_train:
        if not task_id[source] and len(task_train[source]) > 1:
            chosen = min(task_train[source], key=lambda item: _stable_int("id-fallback", item.key))
            task_train[source].remove(chosen)
            task_id[source].append(chosen)

    network_train: dict[str, list[NetworkSample]] = {"synthetic": [], "dagbench": [], "topology_zoo": []}
    network_id: dict[str, list[NetworkSample]] = {"synthetic": [], "dagbench": [], "topology_zoo": []}
    network_ood: list[NetworkSample] = []
    for item in networks:
        if item.source == "topology_zoo" and item.family in heldout_network:
            network_ood.append(item)
        elif _stable_int("network-id-partition", item.key) % 5 == 0:
            network_id[item.source].append(item)
        else:
            network_train[item.source].append(item)
    for source in network_train:
        if not network_id[source] and len(network_train[source]) > 1:
            chosen = min(network_train[source], key=lambda item: _stable_int("id-fallback", item.key))
            network_train[source].remove(chosen)
            network_id[source].append(chosen)
    return {
        "task_train": task_train,
        "task_id": task_id,
        "task_ood": task_ood,
        "network_train": network_train,
        "network_id": network_id,
        "network_ood": network_ood,
    }


def build_mixed_splits(config: Mapping[str, Any], config_dir: Path) -> tuple[dict[str, list[Scenario]], dict[str, Any]]:
    seed = int(config["seed"])
    dataset = config["dataset"]
    catalog = discover_catalog(config, config_dir)
    pools = _partition_catalog(config, catalog)
    counts = dataset.get("split_counts", {"train": 8, "id_validation": 4})
    task_weights = dataset.get("task_source_weights", {"stg": 0.4, "dagbench": 0.4, "synthetic": 0.2})
    network_weights = dataset.get("network_source_weights", {"synthetic": 0.5, "topology_zoo": 0.3, "dagbench": 0.2})
    require_external = bool(dataset.get("require_external_sources", False))
    if require_external:
        missing = [name for name in ("stg", "dagbench_tasks", "topology_zoo") if catalog["discovered"][name] == 0]
        if missing:
            raise FileNotFoundError(f"required external dataset sources are missing: {', '.join(missing)}")

    split_specs = {
        "train": (pools["task_train"], pools["network_train"]),
        "id_validation": (pools["task_id"], pools["network_id"]),
        "dag_ood": ({"dagbench": pools["task_ood"]}, pools["network_id"]),
        "network_ood": (pools["task_id"], {"topology_zoo": pools["network_ood"]}),
        "joint_ood": ({"dagbench": pools["task_ood"]}, {"topology_zoo": pools["network_ood"]}),
    }
    splits: dict[str, list[Scenario]] = {}
    provenance: dict[str, list[dict[str, Any]]] = {}
    for split_index, split in enumerate(SPLIT_NAMES):
        count = int(counts.get(split, 0))
        if count <= 0:
            splits[split] = []
            provenance[split] = []
            continue
        task_pool, network_pool = split_specs[split]
        rng = np.random.default_rng(_stable_int(seed, "split", split))
        task_schedule = _source_schedule(
            rng, task_pool, task_weights, count, context=f"{split} task"
        )
        network_schedule = _source_schedule(
            rng, network_pool, network_weights, count, context=f"{split} network"
        )
        task_orders = {
            name: list(rng.permutation(len(pool))) for name, pool in task_pool.items() if pool
        }
        network_orders = {
            name: list(rng.permutation(len(pool))) for name, pool in network_pool.items() if pool
        }
        task_positions = {name: 0 for name in task_orders}
        network_positions = {name: 0 for name in network_orders}
        scenarios: list[Scenario] = []
        records: list[dict[str, Any]] = []
        for index in range(count):
            task_source = task_schedule[index]
            network_source = network_schedule[index]
            task_candidates = task_pool[task_source]
            network_candidates = network_pool[network_source]
            task_order = task_orders[task_source]
            network_order = network_orders[network_source]
            task = task_candidates[task_order[task_positions[task_source] % len(task_order)]]
            network = network_candidates[
                network_order[network_positions[network_source] % len(network_order)]
            ]
            task_positions[task_source] += 1
            network_positions[network_source] += 1
            scenario_seed = int(_stable_int(seed, split, index, task.key, network.key) % (2**31))
            scenario = assemble_scenario(
                task,
                network,
                scenario_id=f"{split}-{index:04d}",
                seed=scenario_seed,
                max_resource_nodes=(
                    int(dataset["max_resource_nodes"])
                    if dataset.get("max_resource_nodes") is not None
                    else None
                ),
            )
            scenarios.append(scenario)
            records.append(
                {
                    "scenario_id": scenario.id,
                    "scenario_sha256": scenario.content_hash(),
                    "task": {"key": task.key, "source": task.source, "family": task.family, "path": task.path, "raw_sha256": task.raw_sha256},
                    "network": {"key": network.key, "source": network.source, "family": network.family, "path": network.path, "raw_sha256": network.raw_sha256},
                    "augmentation_seed": scenario_seed,
                    "task_count": scenario.task_count,
                    "resource_count_original": len(network.topology.resources),
                    "resource_count_materialized": scenario.resource_count,
                }
            )
        splits[split] = scenarios
        provenance[split] = records
    manifest = {
        "format_version": 1,
        "adapter_version": ADAPTER_VERSION,
        "seed": seed,
        "path_contract": {
            "base": "repository_root",
            "separator": "/",
            "external_inputs": "external-sha256://<digest>/<filename>",
            "absolute_paths_allowed": False,
        },
        "source_revisions": {
            "stg_zenodo_record": "18927122",
            "stg_archive_sha256": "03bc163c13ae8601f8cb20ac1573746a5262c50edbf0c4e9e748968675ea5f7d",
            "dagbench_git_commit": "e69984fcf48f3c66bd9571a9d50591b61de42722",
            "topology_zoo_git_commit": "ae88b808d71e0b1852186aad2b0ad694b6b5c864",
        },
        "units": {"workload": "source cost units", "memory": "MiB", "edge_data": "MB", "bandwidth": "MB/s", "latency": "seconds"},
        "augmentation": {
            "dagbench_missing_task_fields": "deterministic task-requirements-v1",
            "dagbench_zero_cost": "clamped to 1e-6; task and dependencies retained",
            "resource_compatibility": "highest-capability cloud-preferred node expanded so every task has a legal action",
            "execution_time": "workload / (resource speed * task-type affinity)",
            "max_resource_nodes": dataset.get("max_resource_nodes"),
        },
        "discovered": catalog["discovered"],
        "heldout_dag_families": sorted(set(dataset.get("heldout_dag_families", []))),
        "heldout_topologies": sorted(set(dataset.get("heldout_topologies", []))),
        "splits": provenance,
    }
    train_tasks = {row["task"]["key"] for row in provenance["train"]}
    id_tasks = {row["task"]["key"] for row in provenance["id_validation"]}
    train_networks = {row["network"]["key"] for row in provenance["train"]}
    id_networks = {row["network"]["key"] for row in provenance["id_validation"]}
    if train_tasks & id_tasks:
        raise ValueError("task identity leakage between train and id_validation")
    if train_networks & id_networks:
        raise ValueError("network identity leakage between train and id_validation")
    scenario_hashes = [
        row["scenario_sha256"] for records in provenance.values() for row in records
    ]
    if len(scenario_hashes) != len(set(scenario_hashes)):
        raise ValueError("duplicate materialized scenario across mixed-data splits")
    manifest["leakage_audit"] = {
        "train_id_task_key_intersection": 0,
        "train_id_network_key_intersection": 0,
        "duplicate_scenario_hashes": 0,
        "status": "pass",
    }
    return splits, manifest


def materialize_mixed_dataset(
    config_path: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    source = Path(config_path).resolve()
    config = json.loads(source.read_text(encoding="utf-8-sig"))
    destination = Path(output_dir or config["dataset"].get("cache_dir", "outputs/mixed-dataset"))
    destination.mkdir(parents=True, exist_ok=True)
    splits, manifest = build_mixed_splits(config, source.parent)
    for split, scenarios in splits.items():
        split_dir = destination / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for scenario in scenarios:
            scenario.save(split_dir / f"{scenario.id}.json")
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def load_materialized_splits(cache_dir: str | Path) -> dict[str, list[Scenario]]:
    root = Path(cache_dir)
    return {
        split: [Scenario.load(path) for path in sorted((root / split).glob("*.json"))]
        for split in SPLIT_NAMES
    }


def assemble_enhanced_graphs(
    task_graph: EnhancedTaskGraph,
    resource_graph: EnhancedResourceGraph,
    *,
    scenario_id: str,
    seed: int,
    max_resource_nodes: int | None = None,
) -> Scenario:
    """Join independently stored model inputs into the existing Scenario API."""

    placeholder = Scenario(
        id=f"{scenario_id}-task-placeholder",
        seed=task_graph.seed,
        tasks=task_graph.tasks,
        resources=_compatible_topology(task_graph.tasks, resource_graph.topology).resources,
        edges=task_graph.edges,
        bandwidth=resource_graph.topology.bandwidth,
        latency=resource_graph.topology.latency,
    )
    task_sample = TaskSample(
        task_graph.id, task_graph.family, task_graph.source, None, placeholder, None
    )
    network_sample = NetworkSample(
        resource_graph.id,
        resource_graph.family,
        resource_graph.source,
        None,
        resource_graph.topology,
        None,
    )
    return assemble_scenario(
        task_sample,
        network_sample,
        scenario_id=scenario_id,
        seed=seed,
        max_resource_nodes=max_resource_nodes,
    )


def export_separated_enhanced_dataset(
    config_path: str | Path,
    output_dir: str | Path = "data/enhanced-v1",
) -> Path:
    """Export every enhanced task graph and resource graph independently."""

    source = Path(config_path).resolve()
    config = json.loads(source.read_text(encoding="utf-8-sig"))
    catalog = discover_catalog(config, source.parent, prefer_exported=False)
    partition = _partition_catalog(config, catalog)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    task_root = destination / "task_models"
    resource_root = destination / "resource_models"

    task_roles: dict[str, str] = {}
    for pools_name, role in (("task_train", "train_pool"), ("task_id", "id_validation_pool")):
        for pool in partition[pools_name].values():
            for item in pool:
                task_roles[item.key] = role
    for item in partition["task_ood"]:
        task_roles[item.key] = "dag_ood_pool"

    network_roles: dict[str, str] = {}
    for pools_name, role in (("network_train", "train_pool"), ("network_id", "id_validation_pool")):
        for pool in partition[pools_name].values():
            for item in pool:
                network_roles[item.key] = role
    for item in partition["network_ood"]:
        network_roles[item.key] = "network_ood_pool"

    task_entries: list[dict[str, Any]] = []
    for item in sorted(catalog["tasks"], key=lambda value: (value.source, value.key)):
        payload = _task_payload(item)
        directory = task_root / item.source
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / _filename(item.key)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        role = task_roles.get(item.key)
        if role is None and item.source == "stg" and item.family == "stg-test":
            role = "reserved_external_test"
        task_entries.append(
            {
                "id": item.key,
                "source": item.source,
                "family": item.family,
                "role": role or "not_selected",
                "file": target.relative_to(destination).as_posix(),
                "enhanced_sha256": _payload_sha256(payload),
                "raw_sha256": item.raw_sha256,
                "task_count": len(item.scenario.tasks),
                "edge_count": len(item.scenario.edges),
            }
        )

    resource_entries: list[dict[str, Any]] = []
    for item in sorted(catalog["networks"], key=lambda value: (value.source, value.key)):
        payload = _resource_payload(item)
        directory = resource_root / item.source
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / _filename(item.key)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        resource_entries.append(
            {
                "id": item.key,
                "source": item.source,
                "family": item.family,
                "role": network_roles.get(item.key, "not_selected"),
                "file": target.relative_to(destination).as_posix(),
                "enhanced_sha256": _payload_sha256(payload),
                "raw_sha256": item.raw_sha256,
                "resource_count": len(item.topology.resources),
            }
        )

    task_manifest = {
        "format_version": 1,
        "kind": "enhanced_task_graph_collection",
        "adapter_version": ADAPTER_VERSION,
        "count": len(task_entries),
        "sources": dict(sorted({name: sum(row["source"] == name for row in task_entries) for name in {row["source"] for row in task_entries}}.items())),
        "entries": task_entries,
    }
    resource_manifest = {
        "format_version": 1,
        "kind": "enhanced_resource_graph_collection",
        "adapter_version": ADAPTER_VERSION,
        "count": len(resource_entries),
        "sources": dict(sorted({name: sum(row["source"] == name for row in resource_entries) for name in {row["source"] for row in resource_entries}}.items())),
        "entries": resource_entries,
    }
    (task_root / "manifest.json").write_text(
        json.dumps(task_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (resource_root / "manifest.json").write_text(
        json.dumps(resource_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    root_manifest = {
        "format_version": 1,
        "kind": "trisched_separated_enhanced_dataset",
        "adapter_version": ADAPTER_VERSION,
        "config": source.relative_to(_project_root(source.parent)).as_posix(),
        "task_manifest": "task_models/manifest.json",
        "resource_manifest": "resource_models/manifest.json",
        "task_count": len(task_entries),
        "resource_graph_count": len(resource_entries),
        "path_contract": {"base": "this_dataset_directory", "separator": "/", "absolute_paths_allowed": False},
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(root_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest_path
