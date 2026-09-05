from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trisched.data_sources import (
    load_dagbench_json,
    load_stg_json_v2,
    load_topology_zoo_graphml,
    load_topology_zoo_graph,
    scenario_with_topology,
)
from trisched.env import run_policy
from trisched.policies import HeftPolicy
from trisched.scenario import generate_scenario


FIXTURE = Path("tests/fixtures/benchmark/stg_projection_example.json")


def test_constraint_preserving_stg_v2_loader() -> None:
    scenario = load_stg_json_v2(FIXTURE, scenario_id="stg-v2-fixture")
    gpu_task = next(task for task in scenario.tasks if task.accelerator_required)
    assert gpu_task.cpu_cores_required == 1
    assert gpu_task.memory_required == 1024
    assert all(
        scenario.resources[index].has_accelerator
        for index in scenario.compatible_resources(gpu_task.id)
    )
    assert run_policy(scenario, HeftPolicy()).makespan > 0


def test_dagbench_documented_graph_json_loader(tmp_path) -> None:
    payload = {
        "name": "edge.demo",
        "task_graph": {
            "tasks": [{"name": "A", "cost": 8.0}, {"name": "B", "cost": 4.0}],
            "dependencies": [{"source": "A", "target": "B", "size": 2.0}],
        },
        "network": {
            "nodes": [
                {"name": "UE0", "speed": 1.0},
                {"name": "Edge0", "speed": 3.0},
                {"name": "Cloud0", "speed": 8.0},
            ],
            "edges": [
                {"source": "UE0", "target": "Edge0", "speed": 10.0},
                {"source": "Edge0", "target": "Cloud0", "speed": 100.0},
            ],
        },
    }
    source = tmp_path / "graph.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    scenario = load_dagbench_json(source)
    assert all(task.cpu_cores_required >= 1 for task in scenario.tasks)
    assert any(task.memory_required > 0 for task in scenario.tasks)
    assert [resource.kind for resource in scenario.resources] == [
        "device",
        "edge",
        "cloud",
    ]
    assert scenario.bandwidth[0][2] == 10.0
    assert scenario.latency[0][2] == 0.02
    assert run_policy(scenario, HeftPolicy()).makespan > 0


def test_topology_zoo_native_graph_loader(tmp_path) -> None:
    source = tmp_path / "Tiny.graph"
    source.write_text(
        """NODES 3
label x y
N0 0 0
N1 0 0
N2 0 0

EDGES 4
label src dest weight bw delay
L0 0 1 1 20 0.1
L1 1 0 1 20 0.1
L2 1 2 1 80 0.2
L3 2 1 1 80 0.2
""",
        encoding="utf-8",
    )
    topology = load_topology_zoo_graph(source)
    assert len(topology.resources) == 3
    assert topology.bandwidth[0][2] == 20.0
    assert np.isclose(topology.latency[0][2], 0.3)


def test_topology_zoo_graphml_can_be_combined_with_any_dag(tmp_path) -> None:
    graphml = """<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="name" for="node" attr.name="name" attr.type="string"/>
  <key id="bw" for="edge" attr.name="bandwidth" attr.type="double"/>
  <graph edgedefault="undirected">
    <node id="n0"><data key="name">device-a</data></node>
    <node id="n1"><data key="name">edge-a</data></node>
    <node id="n2"><data key="name">cloud-a</data></node>
    <edge source="n0" target="n1"><data key="bw">20</data></edge>
    <edge source="n1" target="n2"><data key="bw">80</data></edge>
  </graph>
</graphml>"""
    source = tmp_path / "tiny.graphml"
    source.write_text(graphml, encoding="utf-8")
    topology = load_topology_zoo_graphml(source)
    scenario = scenario_with_topology(generate_scenario(1, task_count=5), topology)
    assert scenario.resource_count == 3
    assert scenario.bandwidth[0][2] == 20.0
    assert run_policy(scenario, HeftPolicy()).makespan > 0
