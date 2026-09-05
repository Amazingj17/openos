from __future__ import annotations

import json
import hashlib
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .scenario import Edge, Resource, Scenario, Task


@dataclass(frozen=True)
class ResourceTopology:
    """A reusable cloud-edge-device resource graph."""

    resources: tuple[Resource, ...]
    bandwidth: tuple[tuple[float, ...], ...]
    latency: tuple[tuple[float, ...], ...]
    source: str


def _stable_seed(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def enhance_task_requirements(
    scenario: Scenario,
    *,
    seed: int,
) -> Scenario:
    """Deterministically add cloud-edge-device constraints to a plain DAG.

    The original task IDs, workloads, dependency edges and edge data are kept
    byte-for-byte at the value level.  Only fields absent from DAGBench's task
    model are derived.  A sidecar manifest records ``seed`` and this adapter
    version when materialized by :mod:`trisched.mixed_dataset`.
    """

    rng = np.random.default_rng(_stable_seed("task-requirements-v1", seed, scenario.id))
    tasks: list[Task] = []
    for task in scenario.tasks:
        selector = float(rng.random())
        task_type = "gpu" if selector < 0.18 else "data" if selector < 0.48 else "cpu"
        accelerator = task_type == "gpu"
        cores = int(rng.choice((1, 2, 4, 8), p=(0.25, 0.35, 0.3, 0.1)))
        memory_per_core = float(rng.choice((512.0, 1_024.0, 2_048.0)))
        memory = cores * memory_per_core
        features: tuple[str, ...] = ()
        if task_type == "data" and rng.random() < 0.4:
            features = ("high_bandwidth",)
        tasks.append(
            Task(
                id=task.id,
                workload=task.workload,
                cpu_cores_required=cores,
                memory_required=memory,
                accelerator_required=accelerator,
                required_features=features,
                task_type=task_type,
            )
        )
    return Scenario(
        id=scenario.id,
        seed=seed,
        tasks=tuple(tasks),
        resources=scenario.resources,
        edges=scenario.edges,
        bandwidth=scenario.bandwidth,
        latency=scenario.latency,
    )


def topology_from_scenario(scenario: Scenario, *, source: str) -> ResourceTopology:
    """Detach a dataset-provided network so it can be paired with another DAG."""

    return ResourceTopology(
        scenario.resources,
        scenario.bandwidth,
        scenario.latency,
        source,
    )


def _load_json(path: str | Path) -> Mapping[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read dataset JSON {source}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("dataset JSON root must be an object")
    return payload


def _tier_from_name(name: str) -> str | None:
    value = name.lower()
    if any(token in value for token in ("cloud", "datacenter", "server")):
        return "cloud"
    if any(token in value for token in ("edge", "fog", "mec", "gateway")):
        return "edge"
    if any(
        token in value
        for token in ("device", "sensor", "mobile", "client", "ue", "iot")
    ):
        return "device"
    return None


def _assign_tiers(names: Sequence[str], scores: Sequence[float]) -> tuple[str, ...]:
    explicit = [_tier_from_name(name) for name in names]
    unresolved = [index for index, value in enumerate(explicit) if value is None]
    ordered = sorted(unresolved, key=lambda index: (scores[index], names[index]))
    for rank, index in enumerate(ordered):
        position = rank / max(len(ordered) - 1, 1)
        explicit[index] = (
            "device" if position < 0.34 else "edge" if position < 0.67 else "cloud"
        )
    return tuple(str(value) for value in explicit)


def _resource_for_node(
    resource_id: int, name: str, kind: str, original_speed: float
) -> Resource:
    scale = {"device": 1.0, "edge": 4.0, "cloud": 10.0}[kind]
    cores = {"device": 4, "edge": 16, "cloud": 64}[kind]
    # Dataset adapters use MiB.  Synthetic GiB profiles are converted by the
    # mixed-scenario assembler before they are combined with external tasks.
    memory = {"device": 8_192.0, "edge": 65_536.0, "cloud": 262_144.0}[kind]
    accelerator = kind == "cloud" or (kind == "edge" and resource_id % 2 == 0)
    features = {"cpu"}
    if kind == "device":
        features.add("trusted_local")
    else:
        features.add("high_bandwidth")
    if accelerator:
        features.update(("gpu", "accelerator"))
    return Resource(
        id=resource_id,
        name=name,
        kind=kind,
        speed=max(float(original_speed), 1e-9) * scale,
        cpu_cores=cores,
        memory_capacity=memory,
        has_accelerator=accelerator,
        features=tuple(sorted(features)),
    )


def _effective_network(
    node_count: int,
    links: Sequence[tuple[int, int, float, float]],
    *,
    directed: bool,
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
    bandwidth = np.zeros((node_count, node_count), dtype=np.float64)
    latency = np.full((node_count, node_count), np.inf, dtype=np.float64)
    np.fill_diagonal(bandwidth, 1e9)
    np.fill_diagonal(latency, 0.0)
    for source, target, speed, delay in links:
        bandwidth[source, target] = max(bandwidth[source, target], speed)
        latency[source, target] = min(latency[source, target], delay)
        if not directed:
            bandwidth[target, source] = max(bandwidth[target, source], speed)
            latency[target, source] = min(latency[target, source], delay)
    # Widest-path bandwidth and shortest-path latency make sparse physical
    # topologies usable by Scenario's pairwise communication contract.
    for pivot in range(node_count):
        for source in range(node_count):
            for target in range(node_count):
                bandwidth[source, target] = max(
                    bandwidth[source, target],
                    min(bandwidth[source, pivot], bandwidth[pivot, target]),
                )
                latency[source, target] = min(
                    latency[source, target],
                    latency[source, pivot] + latency[pivot, target],
                )
    if np.any(bandwidth <= 0) or not np.all(np.isfinite(latency)):
        raise ValueError("resource network must be strongly connected")
    return (
        tuple(tuple(float(item) for item in row) for row in bandwidth),
        tuple(tuple(float(item) for item in row) for row in latency),
    )


def load_dagbench_json(
    path: str | Path,
    *,
    seed: int = 0,
    scenario_id: str | None = None,
    default_latency: float = 0.01,
) -> Scenario:
    """Load DAGBench's documented SAGA-compatible ``graph.json`` format."""

    payload = _load_json(path)
    task_graph = payload.get("task_graph")
    network = payload.get("network")
    if not isinstance(task_graph, dict) or not isinstance(network, dict):
        raise ValueError("DAGBench JSON requires task_graph and network objects")
    raw_tasks = task_graph.get("tasks")
    raw_dependencies = task_graph.get("dependencies")
    raw_nodes = network.get("nodes")
    raw_links = network.get("edges")
    if not all(isinstance(value, list) for value in (
        raw_tasks, raw_dependencies, raw_nodes, raw_links
    )):
        raise ValueError("DAGBench tasks, dependencies, nodes and edges must be arrays")
    task_names = [str(item["name"]) for item in raw_tasks]
    node_names = [str(item["name"]) for item in raw_nodes]
    if len(set(task_names)) != len(task_names) or len(set(node_names)) != len(node_names):
        raise ValueError("DAGBench task and network node names must be unique")
    task_ids = {name: index for index, name in enumerate(task_names)}
    node_ids = {name: index for index, name in enumerate(node_names)}
    tasks = tuple(
        # Some DAGBench source/sink markers have zero cost.  TriSched requires
        # strictly positive durations, so only those markers receive a tiny,
        # documented epsilon rather than being dropped from the DAG.
        Task(id=index, workload=max(float(item["cost"]), 1e-6), task_type="generic")
        for index, item in enumerate(raw_tasks)
    )
    edges = tuple(
        Edge(
            task_ids[str(item["source"])],
            task_ids[str(item["target"])],
            float(item.get("size", 0.0)),
        )
        for item in raw_dependencies
    )
    node_speeds = [float(item.get("speed", 1.0)) for item in raw_nodes]
    kinds = _assign_tiers(node_names, node_speeds)
    resources = tuple(
        _resource_for_node(index, node_names[index], kinds[index], node_speeds[index])
        for index in range(len(node_names))
    )
    links = [
        (
            node_ids[str(item["source"])],
            node_ids[str(item["target"])],
            float(item["speed"]),
            float(item.get("latency", default_latency)),
        )
        for item in raw_links
    ]
    bandwidth, latency = _effective_network(
        len(resources), links, directed=bool(network.get("directed", False))
    )
    scenario = Scenario(
        id=scenario_id or str(payload.get("name") or Path(path).stem),
        seed=seed,
        tasks=tasks,
        resources=resources,
        edges=edges,
        bandwidth=bandwidth,
        latency=latency,
    )
    return enhance_task_requirements(scenario, seed=seed)


def load_stg_json_v2(
    path: str | Path,
    *,
    scenario_id: str | None = None,
) -> Scenario:
    """Constraint-preserving cloud-edge-device projection of GrapheonRL STG JSON.

    Unlike the frozen benchmark-v1 loader, this opt-in adapter retains cores,
    memory and CPU/GPU feature requirements.  It intentionally uses a new API so
    published v1 manifests and their hashes remain reproducible.
    """

    payload = _load_json(path)
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, dict) or not raw_tasks:
        raise ValueError("STG JSON requires a non-empty tasks object")
    names = sorted(
        raw_tasks,
        key=lambda name: int(name[1:]) if name.startswith("T") else name,
    )
    ids = {name: index for index, name in enumerate(names)}
    all_features = {
        str(feature)
        for item in raw_tasks.values()
        for feature in item.get("features", [])
    }
    resources = (
        Resource(0, "device-cpu", "device", 1.0, 4, 8_192.0, False, ("cpu", "trusted_local")),
        Resource(1, "edge-cpu", "edge", 4.0, 16, 65_536.0, False, ("cpu", "high_bandwidth")),
        Resource(2, "edge-gpu", "edge", 7.0, 24, 65_536.0, True, ("accelerator", "cpu", "gpu", "high_bandwidth")),
        Resource(3, "cloud-cpu", "cloud", 10.0, 64, 262_144.0, False, ("cpu", "high_bandwidth")),
        Resource(4, "cloud-gpu", "cloud", 18.0, 96, 524_288.0, True, tuple(sorted(all_features | {"accelerator", "cpu", "gpu", "high_bandwidth"}))),
    )
    tasks: list[Task] = []
    for name in names:
        item = raw_tasks[name]
        features = tuple(str(value) for value in item.get("features", []))
        accelerator = "gpu" in {value.lower() for value in features}
        tasks.append(
            Task(
                id=ids[name],
                workload=float(item["duration"]),
                cpu_cores_required=int(item.get("cores", 1)),
                memory_required=float(item.get("memory_required", 0.0)),
                accelerator_required=accelerator,
                required_features=tuple(sorted(set(features))),
                task_type="gpu" if accelerator else "cpu",
            )
        )
    edges = tuple(
        Edge(ids[parent], ids[name], float(raw_tasks[parent].get("data", 0.0)))
        for name in names
        for parent in raw_tasks[name].get("dependencies", [])
    )
    kinds = [resource.kind for resource in resources]
    bandwidth: list[list[float]] = []
    latency: list[list[float]] = []
    for source, source_kind in enumerate(kinds):
        bw_row: list[float] = []
        lat_row: list[float] = []
        for target, target_kind in enumerate(kinds):
            if source == target:
                bw_row.append(1e9)
                lat_row.append(0.0)
            elif {source_kind, target_kind} == {"device", "edge"}:
                bw_row.append(100.0)
                lat_row.append(0.01)
            elif {source_kind, target_kind} == {"edge", "cloud"}:
                bw_row.append(10_000.0)
                lat_row.append(0.03)
            else:
                bw_row.append(50.0)
                lat_row.append(0.08)
        bandwidth.append(bw_row)
        latency.append(lat_row)
    meta = payload.get("meta", {})
    stg_info = meta.get("stg_info", {}) if isinstance(meta, dict) else {}
    return Scenario(
        id=scenario_id or f"stg-v2-{Path(path).stem}",
        seed=int(stg_info.get("random_seed", 0)),
        tasks=tuple(tasks),
        resources=resources,
        edges=edges,
        bandwidth=tuple(tuple(row) for row in bandwidth),
        latency=tuple(tuple(row) for row in latency),
    )


def load_topology_zoo_graphml(
    path: str | Path,
    *,
    default_bandwidth: float = 100.0,
    latency_per_hop: float = 0.01,
) -> ResourceTopology:
    """Convert a portable Topology Zoo GraphML graph into a resource graph.

    Pickle variants are deliberately not accepted: GraphML is portable and does
    not execute embedded Python objects.  Compute tiers are a documented,
    deterministic projection from node degree unless a node name states its tier.
    """

    source = Path(path)
    try:
        root = ET.parse(source).getroot()
    except (OSError, ET.ParseError) as error:
        raise ValueError(f"cannot read GraphML topology: {error}") from error
    namespace = root.tag.partition("}")[0].lstrip("{")
    prefix = f"{{{namespace}}}" if namespace else ""
    keys: dict[str, str] = {}
    for key in root.findall(f"{prefix}key"):
        keys[str(key.get("id"))] = str(key.get("attr.name", key.get("id")))
    graph = root.find(f"{prefix}graph")
    if graph is None:
        raise ValueError("GraphML has no graph element")
    raw_nodes = list(graph.findall(f"{prefix}node"))
    node_ids = {str(node.get("id")): index for index, node in enumerate(raw_nodes)}
    names: list[str] = []
    for node in raw_nodes:
        data = {
            keys.get(str(item.get("key")), str(item.get("key"))): item.text
            for item in node.findall(f"{prefix}data")
        }
        names.append(str(data.get("label") or data.get("name") or node.get("id")))
    links: list[tuple[int, int, float, float]] = []
    degrees = [0.0] * len(raw_nodes)
    for edge in graph.findall(f"{prefix}edge"):
        source_id, target_id = str(edge.get("source")), str(edge.get("target"))
        if source_id not in node_ids or target_id not in node_ids:
            raise ValueError("GraphML edge references an unknown node")
        data = {
            keys.get(str(item.get("key")), str(item.get("key"))): item.text
            for item in edge.findall(f"{prefix}data")
        }
        speed = float(
            data.get("LinkSpeed") or data.get("bandwidth") or default_bandwidth
        )
        delay = float(data.get("latency") or latency_per_hop)
        source_index, target_index = node_ids[source_id], node_ids[target_id]
        links.append((source_index, target_index, speed, delay))
        degrees[source_index] += 1.0
        degrees[target_index] += 1.0
    if not raw_nodes or not links:
        raise ValueError("GraphML topology requires nodes and edges")
    kinds = _assign_tiers(names, degrees)
    maximum_degree = max(degrees)
    resources = tuple(
        _resource_for_node(
            index,
            names[index],
            kinds[index],
            1.0 + 0.2 * degrees[index] / maximum_degree,
        )
        for index in range(len(names))
    )
    bandwidth, latency = _effective_network(
        len(resources),
        links,
        directed=str(graph.get("edgedefault", "undirected")) == "directed",
    )
    return ResourceTopology(resources, bandwidth, latency, str(source))


def load_topology_zoo_graph(path: str | Path) -> ResourceTopology:
    """Load the ``NODES``/``EDGES`` text format used by the supplied Zoo fork.

    Traffic-matrix and pickle files are intentionally ignored.  The ``bw`` and
    ``delay`` columns are used when present; duplicate reverse-direction links
    are harmless because the effective network keeps the best direct link.
    """

    source = Path(path)
    try:
        lines = [
            line.strip()
            for line in source.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read Topology Zoo graph {source}: {error}") from error
    if not lines or not lines[0].startswith("NODES "):
        raise ValueError("Topology Zoo graph must start with 'NODES <count>'")
    node_count = int(lines[0].split()[1])
    if node_count <= 0 or len(lines) < node_count + 4:
        raise ValueError("Topology Zoo graph has an invalid node section")
    node_header = lines[1].split()
    node_rows = [lines[index].split() for index in range(2, 2 + node_count)]
    if "label" not in node_header:
        raise ValueError("Topology Zoo node table requires a label column")
    label_index = node_header.index("label")
    names = [row[label_index] for row in node_rows]
    edge_marker = 2 + node_count
    if not lines[edge_marker].startswith("EDGES "):
        raise ValueError("Topology Zoo graph has no EDGES section")
    edge_count = int(lines[edge_marker].split()[1])
    edge_header = lines[edge_marker + 1].split()
    required = {"src", "dest"}
    if not required.issubset(edge_header):
        raise ValueError("Topology Zoo edge table requires src and dest columns")
    rows = [
        lines[index].split()
        for index in range(edge_marker + 2, edge_marker + 2 + edge_count)
    ]
    if len(rows) != edge_count:
        raise ValueError("Topology Zoo graph has a truncated edge section")
    src_index, dest_index = edge_header.index("src"), edge_header.index("dest")
    bw_index = edge_header.index("bw") if "bw" in edge_header else None
    delay_index = edge_header.index("delay") if "delay" in edge_header else None
    degrees = [0.0] * node_count
    links: list[tuple[int, int, float, float]] = []
    for row in rows:
        source_id, target_id = int(row[src_index]), int(row[dest_index])
        if not 0 <= source_id < node_count or not 0 <= target_id < node_count:
            raise ValueError("Topology Zoo edge references an unknown node")
        bandwidth = float(row[bw_index]) if bw_index is not None else 100.0
        delay = float(row[delay_index]) if delay_index is not None else 1.0
        links.append((source_id, target_id, bandwidth, delay))
        degrees[source_id] += 1.0
        degrees[target_id] += 1.0
    kinds = _assign_tiers(names, degrees)
    maximum_degree = max(degrees) or 1.0
    resources = tuple(
        _resource_for_node(
            index,
            f"{source.stem}-{names[index]}",
            kinds[index],
            1.0 + 0.2 * degrees[index] / maximum_degree,
        )
        for index in range(node_count)
    )
    bandwidth, latency = _effective_network(node_count, links, directed=False)
    return ResourceTopology(resources, bandwidth, latency, str(source))


def scenario_with_topology(
    task_scenario: Scenario,
    topology: ResourceTopology,
    *,
    scenario_id: str | None = None,
) -> Scenario:
    """Combine any task DAG with an independently replaceable resource graph."""

    return Scenario(
        id=scenario_id or f"{task_scenario.id}-on-{Path(topology.source).stem}",
        seed=task_scenario.seed,
        tasks=task_scenario.tasks,
        resources=topology.resources,
        edges=task_scenario.edges,
        bandwidth=topology.bandwidth,
        latency=topology.latency,
    )
