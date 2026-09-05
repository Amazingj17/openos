from __future__ import annotations

import json
from pathlib import Path

import pytest

from trisched.mixed_dataset import (
    assemble_enhanced_graphs,
    build_mixed_splits,
    export_separated_enhanced_dataset,
    load_enhanced_resource_graph,
    load_enhanced_task_graph,
    materialize_mixed_dataset,
)


def _write_dag(path: Path, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "task_graph": {
            "tasks": [{"name": "A", "cost": 0.0}, {"name": "B", "cost": 4.0}],
            "dependencies": [{"source": "A", "target": "B", "size": 2.0}],
        },
        "network": {
            "nodes": [
                {"name": "Device0", "speed": 1.0},
                {"name": "Edge0", "speed": 3.0},
                {"name": "Cloud0", "speed": 8.0},
            ],
            "edges": [
                {"source": "Device0", "target": "Edge0", "speed": 10.0},
                {"source": "Edge0", "target": "Cloud0", "speed": 100.0},
            ],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_topology(path: Path) -> None:
    path.write_text(
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


def test_mixed_dataset_has_fixed_id_and_ood_splits(tmp_path) -> None:
    dag_root = tmp_path / "dagbench"
    for index in range(30):
        family = "ood" if index >= 24 else "train"
        _write_dag(dag_root / f"{family}-{index}" / "graph.json", f"{family}.case{index}")
    topology_root = tmp_path / "topologies"
    topology_root.mkdir()
    for name in tuple(f"Train{index}" for index in range(20)) + ("Heldout",):
        _write_topology(topology_root / f"{name}.graph")
    config = {
        "seed": 17,
        "dataset": {
            "mode": "mixed_v1",
            "task_range": [4, 5],
            "resource_count": 3,
            "synthetic_catalog_size": 5,
            "require_external_sources": False,
            "sources": {
                "stg": {"patterns": []},
                "dagbench": {"patterns": [str(dag_root / "**" / "graph.json")]},
                "topology_zoo": {"patterns": [str(topology_root / "*.graph")]},
            },
            "task_source_weights": {"dagbench": 1.0},
            "network_source_weights": {"topology_zoo": 1.0},
            "heldout_dag_families": ["ood"],
            "heldout_topologies": ["heldout"],
            "split_counts": {
                "train": 4,
                "id_validation": 2,
                "dag_ood": 2,
                "network_ood": 2,
                "joint_ood": 2,
            },
        },
    }
    splits, manifest = build_mixed_splits(config, tmp_path)
    assert {name: len(values) for name, values in splits.items()} == {
        "train": 4,
        "id_validation": 2,
        "dag_ood": 2,
        "network_ood": 2,
        "joint_ood": 2,
    }
    assert all(row["task"]["family"] == "ood" for row in manifest["splits"]["dag_ood"])
    assert all(row["network"]["family"] == "heldout" for row in manifest["splits"]["network_ood"])
    train_tasks = {row["task"]["key"] for row in manifest["splits"]["train"]}
    id_tasks = {row["task"]["key"] for row in manifest["splits"]["id_validation"]}
    assert train_tasks.isdisjoint(id_tasks)
    assert min(task.workload for scenario in splits["train"] for task in scenario.tasks) > 0
    paths = [
        item[role]["path"]
        for rows in manifest["splits"].values()
        for item in rows
        for role in ("task", "network")
        if item[role]["path"] is not None
    ]
    assert all("\\" not in path for path in paths)
    assert all(":/Users/" not in path for path in paths)
    assert manifest["path_contract"]["absolute_paths_allowed"] is False

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    manifest_path = materialize_mixed_dataset(config_path, tmp_path / "cache")
    assert manifest_path.is_file()
    assert (tmp_path / "cache" / "joint_ood" / "joint_ood-0000.json").is_file()

    export_root = tmp_path / "enhanced"
    root_manifest_path = export_separated_enhanced_dataset(config_path, export_root)
    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    task_manifest = json.loads(
        (export_root / root_manifest["task_manifest"]).read_text(encoding="utf-8")
    )
    resource_manifest = json.loads(
        (export_root / root_manifest["resource_manifest"]).read_text(encoding="utf-8")
    )
    assert task_manifest["count"] == 35
    assert resource_manifest["count"] == 56
    task_graph = load_enhanced_task_graph(export_root / task_manifest["entries"][0]["file"])
    resource_graph = load_enhanced_resource_graph(
        export_root / resource_manifest["entries"][0]["file"]
    )
    scenario = assemble_enhanced_graphs(
        task_graph, resource_graph, scenario_id="joined", seed=99, max_resource_nodes=3
    )
    assert scenario.task_count == len(task_graph.tasks)
    assert scenario.resource_count == 3
    assert all(scenario.compatible_resources(task.id) for task in scenario.tasks)

    exported_config = json.loads(json.dumps(config))
    exported_config["dataset"]["sources"]["enhanced"] = {
        "prefer": True,
        "manifest": str(root_manifest_path),
    }
    exported_splits, exported_manifest = build_mixed_splits(exported_config, tmp_path)
    original_hashes = [
        scenario.content_hash() for values in splits.values() for scenario in values
    ]
    exported_hashes = [
        scenario.content_hash()
        for values in exported_splits.values()
        for scenario in values
    ]
    assert exported_hashes == original_hashes
    assert exported_manifest["discovered"]["dagbench_tasks"] == 30

    tampered_path = export_root / task_manifest["entries"][0]["file"]
    tampered = json.loads(tampered_path.read_text(encoding="utf-8"))
    tampered["id"] = "tampered"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="enhanced task graph hash mismatch"):
        build_mixed_splits(exported_config, tmp_path)
