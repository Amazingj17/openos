from __future__ import annotations

import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.run_model_comparison import main
from trisched.benchmark import build_stg_manifest
from trisched.learning import TEACHER_FEATURE_NAMES


FIXTURE = (
    Path(__file__).parent / "fixtures" / "benchmark" / "stg_projection_example.json"
)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source_set(root: Path, count: int = 8) -> None:
    base = json.loads(FIXTURE.read_text(encoding="utf-8"))
    root.mkdir(parents=True)
    for index in range(count):
        payload = json.loads(json.dumps(base))
        payload["meta"]["stg_info"]["random_seed"] = 1500 + index
        payload["tasks"]["T1"]["duration"] = 6.0 + index
        (root / f"rand{index:04d}_hetero.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _prepare_comparison_config(tmp_path: Path) -> tuple[Path, Path]:
    raw_root = tmp_path / "raw"
    source = raw_root / "rnc50_hetero"
    _write_source_set(source)
    manifest = build_stg_manifest(
        source,
        {"train": 4, "validation": 2, "test": 2},
    )
    manifest_path = tmp_path / "benchmark.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for entry in manifest["entries"]:
        if entry["split"] == "test":
            (raw_root / entry["source"]).unlink()

    common = {
        "format_version": 1,
        "seeds": [51, 52, 53],
        "output_dir": str(tmp_path / "unused"),
        "benchmark": {
            "manifest": str(manifest_path),
            "raw_root": str(raw_root),
        },
        "features": {"exclude": list(TEACHER_FEATURE_NAMES)},
        "behavior_cloning": {
            "hidden_dim": 8,
            "epochs": 1,
            "learning_rate": 0.004,
            "gradient_clip": 5.0,
            "shuffle_seed_offset": 11,
        },
        "ppo": {
            "epochs": 1,
            "episodes_per_epoch": 4,
            "update_epochs": 1,
            "minibatch_size": 4,
            "actor_learning_rate": 0.0003,
            "value_learning_rate": 0.001,
            "value_hidden_dim": 8,
            "gamma": 1.0,
            "gae_lambda": 0.95,
            "clip_ratio": 0.2,
            "entropy_coefficient": 0.001,
            "target_kl": 0.03,
            "gradient_clip": 5.0,
            "shuffle_seed_offset": 17,
        },
        "selection": {
            "metric": "validation_mean_ratio",
            "failure_penalty_ratio": 7.0,
            "target_ratio": 10.0,
        },
    }
    mlp = {
        **common,
        "ablation": {"teacher_feature_reference_seed": 51},
    }
    task_gnn = {
        **common,
        "task_gnn": {
            "architecture": "task_gnn_v1",
            "message_dim": 4,
        },
    }
    mlp_config = tmp_path / "mlp.json"
    task_gnn_config = tmp_path / "task_gnn.json"
    mlp_config.write_text(json.dumps(mlp), encoding="utf-8")
    task_gnn_config.write_text(json.dumps(task_gnn), encoding="utf-8")
    output = tmp_path / "comparison-output"
    pipeline_config = {
        "format_version": 1,
        "output_dir": str(output),
        "masked_mlp_config": str(mlp_config),
        "task_gnn_config": str(task_gnn_config),
        "comparison": {
            "bootstrap_samples": 200,
            "bootstrap_seed": 51,
            "latency_repeats": 0,
        },
    }
    pipeline_config_path = tmp_path / "comparison.json"
    pipeline_config_path.write_text(json.dumps(pipeline_config), encoding="utf-8")
    return pipeline_config_path, output


def test_one_click_model_comparison_trains_validates_and_visualizes(
    tmp_path: Path,
) -> None:
    config_path, output = _prepare_comparison_config(tmp_path)

    assert main(["--config", str(config_path)]) == 0

    expected = {
        "resolved_comparison_config.json",
        "comparison_pipeline_summary.json",
        "comparison_pipeline_manifest.json",
        "masked_mlp/ppo_summary.json",
        "masked_mlp/ppo_run_manifest.json",
        "task_gnn/task_gnn_summary.json",
        "task_gnn/task_gnn_run_manifest.json",
        "results/comparison.json",
        "results/comparison.html",
        "results/comparison.svg",
        "results/comparison_per_instance.csv",
        "results/comparison_per_seed.csv",
        "results/comparison_per_scenario.csv",
        "results/comparison_manifest.json",
    }
    for name in expected:
        assert (output / name).is_file(), name

    summary = json.loads(
        (output / "comparison_pipeline_summary.json").read_text(encoding="utf-8")
    )
    assert summary["mode"] == "masked_mlp_task_gnn_train_validate_compare"
    assert summary["test_accessed"] is False
    assert summary["paired_validation"]["seed_count"] == 3
    assert summary["paired_validation"]["scenario_count"] == 2

    comparison = json.loads(
        (output / "results" / "comparison.json").read_text(encoding="utf-8")
    )
    assert comparison["inputs"]["test_accessed"] is False
    assert comparison["visualization"] == {
        "html": "comparison.html",
        "svg": "comparison.svg",
    }
    assert len(comparison["per_seed"]) == 3
    assert len(comparison["per_scenario_seed_mean"]) == 2

    with (output / "results" / "comparison_per_instance.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        assert len(list(csv.DictReader(handle))) == 6
    with (output / "results" / "comparison_per_seed.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        assert len(list(csv.DictReader(handle))) == 3
    with (output / "results" / "comparison_per_scenario.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        assert len(list(csv.DictReader(handle))) == 2

    ET.parse(output / "results" / "comparison.svg")
    html_text = (output / "results" / "comparison.html").read_text(encoding="utf-8")
    assert "Masked MLP 与 Task-GNN 性能比较" in html_text
    assert "comparison.svg" in html_text
    assert "公开 test 未访问" in html_text

    for manifest_name in (
        "results/comparison_manifest.json",
        "comparison_pipeline_manifest.json",
    ):
        manifest = json.loads((output / manifest_name).read_text(encoding="utf-8"))
        manifest_root = (output / manifest_name).parent
        if manifest_name == "comparison_pipeline_manifest.json":
            manifest_root = output
        for name, metadata in manifest["artifacts"].items():
            artifact = manifest_root / name
            assert artifact.stat().st_size == metadata["bytes"]
            assert _file_hash(artifact) == metadata["sha256"]

    assert main(["--config", str(config_path)]) == 2
