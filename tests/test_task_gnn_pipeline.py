from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import trisched.ppo as ppo_module
from trisched.bc import BehaviorCloningError
from trisched.benchmark import build_stg_manifest
from trisched.cli import main
from trisched.gnn import TASK_GNN_FEATURE_NAMES
from trisched.learning import TEACHER_FEATURE_NAMES
from trisched.ppo import load_task_gnn_config, run_task_gnn_pipeline


FIXTURE = (
    Path(__file__).parent / "fixtures" / "benchmark" / "stg_projection_example.json"
)


def _write_source_set(root: Path, count: int = 8) -> None:
    base = json.loads(FIXTURE.read_text(encoding="utf-8"))
    root.mkdir(parents=True)
    for index in range(count):
        payload = json.loads(json.dumps(base))
        payload["meta"]["stg_info"]["random_seed"] = 1200 + index
        payload["tasks"]["T1"]["duration"] = 5.0 + index
        (root / f"rand{index:04d}_hetero.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _prepare_config(tmp_path: Path, *, ppo_epochs: int = 1) -> Path:
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
    config = {
        "format_version": 1,
        "seeds": [41, 42, 43],
        "output_dir": str(tmp_path / "unused"),
        "benchmark": {
            "manifest": str(manifest_path),
            "raw_root": str(raw_root),
        },
        "features": {"exclude": list(TEACHER_FEATURE_NAMES)},
        "task_gnn": {
            "architecture": "task_gnn_v1",
            "message_dim": 4,
        },
        "behavior_cloning": {
            "hidden_dim": 8,
            "epochs": 1,
            "learning_rate": 0.004,
            "gradient_clip": 5.0,
            "shuffle_seed_offset": 11,
        },
        "ppo": {
            "epochs": ppo_epochs,
            "episodes_per_epoch": 4,
            "update_epochs": 2,
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
    config_path = tmp_path / "task_gnn_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _directory_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {name: np.asarray(values[name]) for name in values.files}


def _assert_npz_equal(first: Path, second: Path) -> None:
    first_arrays = _npz_arrays(first)
    second_arrays = _npz_arrays(second)
    assert set(first_arrays) == set(second_arrays)
    for name in first_arrays:
        assert np.array_equal(first_arrays[name], second_arrays[name]), name


def _transaction_residue(root: Path, output_name: str) -> list[Path]:
    return sorted(root.glob(f".{output_name}.task-gnn-resume-*"))


def test_task_gnn_config_rejects_unreviewed_variables_and_schema_drift(
    tmp_path: Path,
) -> None:
    config_path = _prepare_config(tmp_path)
    original = json.loads(config_path.read_text(encoding="utf-8"))
    injections = (
        ("resource-GNN", {**original, "resource_gnn": {"enabled": True}}),
        (
            "curriculum",
            {
                **original,
                "ppo": {**original["ppo"], "curriculum": "small_to_large"},
            },
        ),
    )
    for _, payload in injections:
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(BehaviorCloningError) as captured:
            load_task_gnn_config(config_path)
        assert captured.value.code == "task_gnn_config_variable"

    drifted = json.loads(json.dumps(original))
    drifted["features"]["exclude"].append("progress")
    config_path.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(BehaviorCloningError) as captured:
        load_task_gnn_config(config_path)
    assert captured.value.code == "task_gnn_feature_schema"


def test_task_gnn_cli_writes_complete_test_free_evidence(tmp_path: Path) -> None:
    config_path = _prepare_config(tmp_path)
    output = tmp_path / "task-gnn-output"

    assert (
        main(
            [
                "train-task-gnn",
                "--config",
                str(config_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    summary = json.loads((output / "task_gnn_summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "stg_task_gnn_ppo"
    assert summary["data_access"]["loaded_splits"] == ["train", "validation"]
    assert summary["data_access"]["test_accessed"] is False
    assert summary["architecture"]["architecture"] == "task_gnn_v1"
    assert summary["architecture"]["base_feature_names"] == list(TASK_GNN_FEATURE_NAMES)
    assert summary["aggregate_validation"]["seed_count"] == 3
    manifest = json.loads(
        (output / "task_gnn_run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["execution"] == {
        "resume_requested": False,
        "resumed_seeds": [],
        "resume_boundary": "completed_ppo_epoch",
        "publication_mode": "direct_new_directory",
    }
    assert manifest["inputs"]["test_accessed"] is False
    assert set(manifest["inputs"]["splits"]) == {"train", "validation"}
    assert len(manifest["artifacts"]) == 32
    for name, metadata in manifest["artifacts"].items():
        artifact = output / name
        assert artifact.stat().st_size == metadata["bytes"]
        assert _file_hash(artifact) == metadata["sha256"]


def test_task_gnn_resume_matches_continuous_and_rolls_back_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _prepare_config(tmp_path, ppo_epochs=2)
    continuous = tmp_path / "continuous"
    resumed = tmp_path / "resumed"
    continuous_summary_path = run_task_gnn_pipeline(config_path, continuous)

    original_update = ppo_module._update_task_gnn_ppo
    update_calls = 0

    def interrupt_during_second_epoch(*args, **kwargs):
        nonlocal update_calls
        update_calls += 1
        if update_calls == 2:
            raise RuntimeError("simulated task-GNN interruption")
        return original_update(*args, **kwargs)

    monkeypatch.setattr(
        ppo_module,
        "_update_task_gnn_ppo",
        interrupt_during_second_epoch,
    )
    with pytest.raises(RuntimeError, match="simulated task-GNN interruption"):
        run_task_gnn_pipeline(config_path, resumed)
    interrupted_state = resumed / "seed_41_task_gnn_ppo_training_state.npz"
    assert interrupted_state.is_file()
    with np.load(interrupted_state, allow_pickle=False) as state:
        metadata = json.loads(str(state["metadata_json"].item()))
    assert metadata["completed_epoch"] == 1
    assert metadata["test_accessed"] is False

    monkeypatch.setattr(ppo_module, "_update_task_gnn_ppo", original_update)
    resumed_summary_path = run_task_gnn_pipeline(
        config_path,
        resumed,
        resume=True,
    )
    assert json.loads(resumed_summary_path.read_text(encoding="utf-8")) == (
        json.loads(continuous_summary_path.read_text(encoding="utf-8"))
    )
    for seed in (41, 42, 43):
        prefix = f"seed_{seed}_task_gnn"
        assert json.loads(
            (resumed / f"{prefix}_training_curve.json").read_text(encoding="utf-8")
        ) == json.loads(
            (continuous / f"{prefix}_training_curve.json").read_text(encoding="utf-8")
        )
        for suffix in (
            "ppo_best_policy.npz",
            "ppo_last_policy.npz",
            "ppo_best_value.npz",
            "ppo_last_value.npz",
            "ppo_training_state.npz",
        ):
            _assert_npz_equal(
                resumed / f"{prefix}_{suffix}",
                continuous / f"{prefix}_{suffix}",
            )
    resumed_manifest = json.loads(
        (resumed / "task_gnn_run_manifest.json").read_text(encoding="utf-8")
    )
    assert resumed_manifest["execution"] == {
        "resume_requested": True,
        "resumed_seeds": [41],
        "resume_boundary": "completed_ppo_epoch",
        "publication_mode": "staging_directory_swap",
    }

    before_failure = _directory_snapshot(resumed)
    original_write_json = ppo_module._write_json

    def fail_after_first_seed(path, value):
        if Path(path).name == "seed_42_task_gnn_training_curve.json":
            raise RuntimeError("simulated post-seed task-GNN failure")
        return original_write_json(path, value)

    monkeypatch.setattr(ppo_module, "_write_json", fail_after_first_seed)
    with pytest.raises(RuntimeError, match="simulated post-seed task-GNN failure"):
        run_task_gnn_pipeline(config_path, resumed, resume=True)
    assert _directory_snapshot(resumed) == before_failure
    assert _transaction_residue(tmp_path, resumed.name) == []
