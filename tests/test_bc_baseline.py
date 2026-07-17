from __future__ import annotations

import hashlib
import json
from pathlib import Path

from trisched.bc import run_bc_pipeline
from trisched.benchmark import (
    build_stg_manifest,
    load_frozen_split,
)
from trisched.env import run_policy, validate_schedule
from trisched.learning import MaskedMLPPolicy


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "benchmark"
    / "stg_projection_example.json"
)


def _write_source_set(root: Path, count: int = 6) -> None:
    base = json.loads(FIXTURE.read_text(encoding="utf-8"))
    root.mkdir(parents=True)
    for index in range(count):
        payload = json.loads(json.dumps(base))
        payload["meta"]["stg_info"]["random_seed"] = 700 + index
        payload["tasks"]["T1"]["duration"] = 4.0 + index
        (root / f"rand{index:04d}_hetero.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _prepare_config(tmp_path: Path) -> tuple[Path, Path, Path]:
    raw_root = tmp_path / "raw"
    source = raw_root / "rnc50_hetero"
    _write_source_set(source)
    manifest = build_stg_manifest(
        source,
        {"train": 4, "validation": 1, "test": 1},
    )
    manifest_path = tmp_path / "benchmark.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    config = {
        "format_version": 1,
        "seed": 17,
        "output_dir": str(tmp_path / "unused"),
        "benchmark": {
            "manifest": str(manifest_path),
            "raw_root": str(raw_root),
        },
        "training": {
            "hidden_dim": 8,
            "epochs": 2,
            "learning_rate": 0.004,
            "gradient_clip": 5.0,
            "shuffle_seed_offset": 11,
        },
        "selection": {
            "metric": "validation_mean_ratio",
            "failure_penalty_ratio": 7.0,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path, manifest_path, raw_root


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_teacher_bc_pipeline_is_reproducible_and_test_free(
    tmp_path: Path,
) -> None:
    config_path, manifest_path, raw_root = _prepare_config(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_summary_path = run_bc_pipeline(config_path, first)
    second_summary_path = run_bc_pipeline(config_path, second)
    first_summary = json.loads(first_summary_path.read_text(encoding="utf-8"))
    second_summary = json.loads(second_summary_path.read_text(encoding="utf-8"))

    assert first_summary["data_access"] == {
        "loaded_splits": ["train", "validation"],
        "teacher_split": "train",
        "model_selection_split": "validation",
        "test_accessed": False,
        "test_status": "forbidden_until_final_evaluation",
    }
    assert first_summary["teacher"]["scenario_count"] == 4
    assert first_summary["teacher"]["action_count"] == 12
    assert first_summary["teacher"]["failure_count"] == 0
    assert first_summary["best_validation"]["scenario_count"] == 1
    assert first_summary["best_validation"]["failure_count"] == 0
    assert first_summary["best_validation"]["illegal_action_count"] == 0
    assert first_summary["best_validation"]["illegal_action_rate"] == 0.0
    assert first_summary["publishable"] is True
    assert first_summary["best_checkpoint"]["parameter_sha256"] == (
        second_summary["best_checkpoint"]["parameter_sha256"]
    )
    assert first_summary["last_checkpoint"]["parameter_sha256"] == (
        second_summary["last_checkpoint"]["parameter_sha256"]
    )

    for name in (
        "train_teacher_manifest.json",
        "validation_reference_manifest.json",
        "bc_training_curve.json",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert (first / "teacher_failures.jsonl").read_text(encoding="utf-8") == ""
    assert (first / "validation_failures.jsonl").read_text(
        encoding="utf-8"
    ) == ""

    teacher = json.loads(
        (first / "train_teacher_manifest.json").read_text(encoding="utf-8")
    )
    reference = json.loads(
        (first / "validation_reference_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert teacher["split"] == "train"
    assert teacher["purpose"] == "behavior_cloning_teacher"
    assert teacher["scenario_count"] == 4
    assert all(
        entry["production_independent_match"] for entry in teacher["entries"]
    )
    assert reference["split"] == "validation"
    assert reference["purpose"] == "model_selection_reference"
    assert not ({entry["scenario_id"] for entry in teacher["entries"]} & {
        entry["scenario_id"] for entry in reference["entries"]
    })

    validation = load_frozen_split(
        raw_root,
        manifest_path,
        "validation",
        purpose="evaluation",
    )
    policy = MaskedMLPPolicy.load(first / "bc_best.npz")
    result = run_policy(validation[0], policy)
    validate_schedule(validation[0], result)

    run_manifest = json.loads(
        (first / "bc_run_manifest.json").read_text(encoding="utf-8")
    )
    assert run_manifest["inputs"]["test_accessed"] is False
    assert set(run_manifest["inputs"]["splits"]) == {"train", "validation"}
    for name, metadata in run_manifest["artifacts"].items():
        artifact = first / name
        assert artifact.stat().st_size == metadata["bytes"]
        assert _file_hash(artifact) == metadata["sha256"]
