from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import trisched.ppo as ppo_module
from trisched.bc import BehaviorCloningError
from trisched.benchmark import build_stg_manifest, load_frozen_split
from trisched.cli import main
from trisched.env import run_policy, validate_schedule
from trisched.learning import TEACHER_FEATURE_NAMES, MaskedMLPPolicy
from trisched.ppo import ValueNetwork, compute_gae, load_ppo_config, run_ppo_pipeline


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "benchmark"
    / "stg_projection_example.json"
)


def _write_source_set(root: Path, count: int = 8) -> None:
    base = json.loads(FIXTURE.read_text(encoding="utf-8"))
    root.mkdir(parents=True)
    for index in range(count):
        payload = json.loads(json.dumps(base))
        payload["meta"]["stg_info"]["random_seed"] = 900 + index
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
        "seeds": [31, 32, 33],
        "output_dir": str(tmp_path / "unused"),
        "benchmark": {
            "manifest": str(manifest_path),
            "raw_root": str(raw_root),
        },
        "features": {
            "exclude": list(TEACHER_FEATURE_NAMES),
        },
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
        "ablation": {
            "teacher_feature_reference_seed": 31,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path, manifest_path, raw_root


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _directory_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _resume_transaction_residue(root: Path, output_name: str) -> list[Path]:
    return sorted(root.glob(f".{output_name}.ppo-resume-*"))


def test_gae_terminal_bootstrap_and_reward_identity() -> None:
    advantages, returns = compute_gae(
        [0.0, -0.25, -0.75],
        [0.0, 0.0, 0.0],
        gamma=1.0,
        gae_lambda=1.0,
    )
    assert np.allclose(advantages, [-1.0, -1.0, -0.75])
    assert np.allclose(returns, advantages)
    assert float(np.sum([0.0, -0.25, -0.75])) == -1.0


def test_actor_and_value_gradients_match_centered_finite_differences() -> None:
    rng = np.random.default_rng(5)
    feature_names = tuple(
        name
        for name in MaskedMLPPolicy().feature_names
        if name not in TEACHER_FEATURE_NAMES
    )
    features = rng.normal(size=(7, len(feature_names)))
    actor = MaskedMLPPolicy(
        hidden_dim=5,
        seed=3,
        feature_names=feature_names,
    )
    cache = actor.distribution_from_features(features)
    actor_checks = (
        (
            lambda: float(
                np.log(
                    actor.distribution_from_features(features).probabilities[2]
                    + 1e-12
                )
            ),
            actor.log_probability_gradients(cache, 2),
        ),
        (
            lambda: float(
                -np.sum(
                    actor.distribution_from_features(features).probabilities
                    * np.log(
                        actor.distribution_from_features(features).probabilities
                        + 1e-12
                    )
                )
            ),
            actor.entropy_gradients(cache),
        ),
    )
    for objective, analytic in actor_checks:
        for name, values in actor.params.items():
            for index in np.ndindex(values.shape):
                original = values[index]
                values[index] = original + 1e-6
                plus = objective()
                values[index] = original - 1e-6
                minus = objective()
                values[index] = original
                numeric = (plus - minus) / 2e-6
                assert numeric == pytest.approx(
                    analytic[name][index],
                    abs=5e-9,
                )

    critic = ValueNetwork(feature_dim=len(feature_names), hidden_dim=4, seed=7)
    state = critic.state_features(features)
    _, _, analytic = critic.loss_gradients(state, -0.8)
    for name, values in critic.params.items():
        for index in np.ndindex(values.shape):
            original = values[index]
            values[index] = original + 1e-6
            plus_value = critic.predict(state)
            values[index] = original - 1e-6
            minus_value = critic.predict(state)
            values[index] = original
            plus = 0.5 * (plus_value + 0.8) ** 2
            minus = 0.5 * (minus_value + 0.8) ** 2
            numeric = (plus - minus) / 2e-6
            assert numeric == pytest.approx(analytic[name][index], abs=5e-9)


def test_ppo_config_rejects_direct_teacher_feature_leakage(
    tmp_path: Path,
) -> None:
    config_path, _, _ = _prepare_config(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["features"]["exclude"] = ["is_heft_task"]
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BehaviorCloningError) as captured:
        load_ppo_config(config_path)
    assert captured.value.code == "ppo_teacher_feature_leakage"


def test_masked_ppo_pipeline_is_reproducible_multiseed_and_test_free(
    tmp_path: Path,
) -> None:
    config_path, manifest_path, raw_root = _prepare_config(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_summary_path = run_ppo_pipeline(config_path, first)
    second_summary_path = run_ppo_pipeline(config_path, second)
    first_summary = json.loads(first_summary_path.read_text(encoding="utf-8"))
    second_summary = json.loads(second_summary_path.read_text(encoding="utf-8"))

    assert first_summary["mode"] == "stg_masked_ppo"
    assert first_summary["data_access"]["loaded_splits"] == [
        "train",
        "validation",
    ]
    assert first_summary["data_access"]["test_accessed"] is False
    assert first_summary["aggregate_validation"]["seed_count"] == 3
    assert first_summary["validation_gate_passed"] is True
    assert not (
        set(first_summary["features"]["selected"])
        & set(TEACHER_FEATURE_NAMES)
    )
    assert [
        result["best_checkpoint"]["actor"]["parameter_sha256"]
        for result in first_summary["seeds"]
    ] == [
        result["best_checkpoint"]["actor"]["parameter_sha256"]
        for result in second_summary["seeds"]
    ]

    validation = load_frozen_split(
        raw_root,
        manifest_path,
        "validation",
        purpose="evaluation",
    )
    for result in first_summary["seeds"]:
        actor_path = first / result["best_checkpoint"]["actor"]["name"]
        value_path = first / result["best_checkpoint"]["value"]["name"]
        actor = MaskedMLPPolicy.load(actor_path)
        critic = ValueNetwork.load(value_path)
        assert tuple(actor.feature_names) == tuple(
            first_summary["features"]["selected"]
        )
        assert critic.feature_dim == len(actor.feature_names)
        schedule = run_policy(validation[0], actor)
        validate_schedule(validation[0], schedule)

    run_manifest = json.loads(
        (first / "ppo_run_manifest.json").read_text(encoding="utf-8")
    )
    assert run_manifest["execution"]["publication_mode"] == (
        "direct_new_directory"
    )
    assert run_manifest["inputs"]["test_accessed"] is False
    assert set(run_manifest["inputs"]["splits"]) == {"train", "validation"}
    for name, metadata in run_manifest["artifacts"].items():
        artifact = first / name
        assert artifact.stat().st_size == metadata["bytes"]
        assert _file_hash(artifact) == metadata["sha256"]


def test_ppo_seed_extension_inherits_verified_prefix_and_trains_only_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, _, _ = _prepare_config(tmp_path)
    source = tmp_path / "source-three-seed"
    run_ppo_pipeline(config_path, source)
    source_manifest_path = source / "ppo_run_manifest.json"
    source_manifest_sha256 = _file_hash(source_manifest_path)

    extension_config = json.loads(config_path.read_text(encoding="utf-8"))
    extension_config["seeds"] = [31, 32, 33, 34, 35]
    extension_config["output_dir"] = str(tmp_path / "five-seed")
    extension_config["seed_extension"] = {
        "source_dir": str(source),
        "run_manifest_sha256": source_manifest_sha256,
        "reuse_seeds": [31, 32, 33],
    }
    extension_config_path = tmp_path / "extension-config.json"
    extension_config_path.write_text(
        json.dumps(extension_config),
        encoding="utf-8",
    )

    original_train = ppo_module.train_masked_ppo
    trained_seeds: list[int] = []

    def record_train(*args, **kwargs):
        trained_seeds.append(int(kwargs["seed"]))
        return original_train(*args, **kwargs)

    monkeypatch.setattr(ppo_module, "train_masked_ppo", record_train)
    output = tmp_path / "five-seed"
    summary_path = run_ppo_pipeline(extension_config_path, output)
    assert trained_seeds == [34, 35]

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert [result["seed"] for result in summary["seeds"]] == [
        31,
        32,
        33,
        34,
        35,
    ]
    assert summary["aggregate_validation"]["seed_count"] == 5
    assert summary["seed_extension"] == {
        "source_run_manifest": "seed_extension_source_run_manifest.json",
        "source_run_manifest_sha256": source_manifest_sha256,
        "inherited_seeds": [31, 32, 33],
        "trained_seeds": [34, 35],
        "source_test_accessed": False,
    }
    for result in summary["seeds"][:3]:
        assert result["seed_extension"]["mode"] == "inherited_verified"
    for result in summary["seeds"][3:]:
        assert result["seed_extension"]["mode"] == "trained_current_run"

    for seed in (31, 32, 33):
        for suffix in (
            "bc_warm_start.npz",
            "ppo_best_policy.npz",
            "ppo_best_value.npz",
            "ppo_last_policy.npz",
            "ppo_last_value.npz",
            "ppo_training_state.npz",
            "training_curve.json",
            "validation_diagnostics.json",
            "validation_failures.jsonl",
        ):
            name = f"seed_{seed}_{suffix}"
            assert (output / name).read_bytes() == (source / name).read_bytes()
    assert (
        output / "seed_extension_source_run_manifest.json"
    ).read_bytes() == source_manifest_path.read_bytes()

    run_manifest = json.loads(
        (output / "ppo_run_manifest.json").read_text(encoding="utf-8")
    )
    assert run_manifest["execution"]["seed_extension"] == {
        "source_run_manifest": "seed_extension_source_run_manifest.json",
        "source_run_manifest_sha256": source_manifest_sha256,
        "inherited_seeds": [31, 32, 33],
        "trained_seeds": [34, 35],
    }
    assert run_manifest["inputs"]["test_accessed"] is False
    for name, metadata in run_manifest["artifacts"].items():
        artifact = output / name
        assert artifact.stat().st_size == metadata["bytes"]
        assert _file_hash(artifact) == metadata["sha256"]

    inherited_actor = source / "seed_31_ppo_best_policy.npz"
    inherited_actor.write_bytes(inherited_actor.read_bytes() + b"tampered")
    with pytest.raises(BehaviorCloningError) as tampered:
        run_ppo_pipeline(
            extension_config_path,
            tmp_path / "rejected-extension",
        )
    assert tampered.value.code == "ppo_seed_extension_artifact"
    assert not (tmp_path / "rejected-extension").exists()


def test_ppo_epoch_resume_matches_uninterrupted_and_rejects_bad_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, _, _ = _prepare_config(tmp_path)
    original_config_text = config_path.read_text(encoding="utf-8")
    config = json.loads(original_config_text)
    config["ppo"]["epochs"] = 2
    config_path.write_text(json.dumps(config), encoding="utf-8")
    original_config_text = config_path.read_text(encoding="utf-8")

    uninterrupted = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    uninterrupted_summary_path = run_ppo_pipeline(config_path, uninterrupted)

    original_update = ppo_module._update_ppo
    update_calls = 0

    def interrupt_during_second_epoch(*args, **kwargs):
        nonlocal update_calls
        update_calls += 1
        if update_calls == 2:
            raise RuntimeError("simulated interruption")
        return original_update(*args, **kwargs)

    monkeypatch.setattr(
        ppo_module,
        "_update_ppo",
        interrupt_during_second_epoch,
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_ppo_pipeline(config_path, resumed)
    state_path = resumed / "seed_31_ppo_training_state.npz"
    assert state_path.is_file()
    with np.load(state_path, allow_pickle=False) as state:
        metadata = json.loads(str(state["metadata_json"].item()))
    assert metadata["completed_epoch"] == 1
    assert metadata["test_accessed"] is False

    monkeypatch.setattr(ppo_module, "_update_ppo", original_update)
    resumed_summary_path = run_ppo_pipeline(config_path, resumed, resume=True)
    uninterrupted_summary = json.loads(
        uninterrupted_summary_path.read_text(encoding="utf-8")
    )
    resumed_summary = json.loads(resumed_summary_path.read_text(encoding="utf-8"))
    assert resumed_summary == uninterrupted_summary
    for seed in (31, 32, 33):
        assert json.loads(
            (resumed / f"seed_{seed}_training_curve.json").read_text(
                encoding="utf-8"
            )
        ) == json.loads(
            (uninterrupted / f"seed_{seed}_training_curve.json").read_text(
                encoding="utf-8"
            )
        )
        assert (
            resumed / f"seed_{seed}_ppo_training_state.npz"
        ).read_bytes() == (
            uninterrupted / f"seed_{seed}_ppo_training_state.npz"
        ).read_bytes()
    resumed_manifest = json.loads(
        (resumed / "ppo_run_manifest.json").read_text(encoding="utf-8")
    )
    assert resumed_manifest["execution"] == {
        "resume_requested": True,
        "resumed_seeds": [31],
        "resume_boundary": "completed_ppo_epoch",
        "publication_mode": "staging_directory_swap",
    }
    assert len(resumed_manifest["artifacts"]) == 34

    seed_31_state = resumed / "seed_31_ppo_training_state.npz"
    seed_32_state = resumed / "seed_32_ppo_training_state.npz"
    valid_seed_32_bytes = seed_32_state.read_bytes()
    seed_32_state.write_bytes(seed_31_state.read_bytes())
    before_mismatch = _directory_snapshot(resumed)
    with pytest.raises(BehaviorCloningError) as wrong_seed:
        run_ppo_pipeline(config_path, resumed, resume=True)
    assert wrong_seed.value.code == "ppo_resume_state_mismatch"
    assert _directory_snapshot(resumed) == before_mismatch
    assert _resume_transaction_residue(tmp_path, resumed.name) == []
    seed_32_state.write_bytes(valid_seed_32_bytes)
    for name, metadata in resumed_manifest["artifacts"].items():
        artifact = resumed / name
        assert artifact.stat().st_size == metadata["bytes"]
        assert _file_hash(artifact) == metadata["sha256"]

    before_runtime_failure = _directory_snapshot(resumed)
    original_write_json = ppo_module._write_json

    def fail_after_seed_31_writes(path, value):
        if Path(path).name == "seed_32_training_curve.json":
            raise RuntimeError("simulated failure after seed 31 writes")
        return original_write_json(path, value)

    monkeypatch.setattr(ppo_module, "_write_json", fail_after_seed_31_writes)
    with pytest.raises(RuntimeError, match="simulated failure after seed 31 writes"):
        run_ppo_pipeline(config_path, resumed, resume=True)
    assert _directory_snapshot(resumed) == before_runtime_failure
    assert _resume_transaction_residue(tmp_path, resumed.name) == []
    monkeypatch.setattr(ppo_module, "_write_json", original_write_json)

    teacher_path = resumed / "train_teacher_manifest.json"
    original_teacher_bytes = teacher_path.read_bytes()
    changed_teacher = json.loads(original_teacher_bytes.decode("utf-8"))
    changed_teacher["injected_change"] = True
    teacher_path.write_text(json.dumps(changed_teacher), encoding="utf-8")
    with pytest.raises(BehaviorCloningError) as teacher_mismatch:
        run_ppo_pipeline(config_path, resumed, resume=True)
    assert teacher_mismatch.value.code == "ppo_resume_artifact_mismatch"
    teacher_path.write_bytes(original_teacher_bytes)

    valid_state_bytes = state_path.read_bytes()
    state_path.unlink()
    with pytest.raises(BehaviorCloningError) as missing_state:
        run_ppo_pipeline(config_path, resumed, resume=True)
    assert missing_state.value.code == "ppo_resume_state_missing"
    state_path.write_bytes(valid_state_bytes)

    changed_config = json.loads(original_config_text)
    changed_config["ppo"]["actor_learning_rate"] = 0.0004
    config_path.write_text(json.dumps(changed_config), encoding="utf-8")
    with pytest.raises(BehaviorCloningError) as mismatch:
        run_ppo_pipeline(config_path, resumed, resume=True)
    assert mismatch.value.code == "ppo_resume_config_mismatch"

    config_path.write_text(original_config_text, encoding="utf-8")
    with np.load(state_path, allow_pickle=False) as state:
        altered_state = {name: np.asarray(state[name]) for name in state.files}
    altered_metadata = json.loads(str(altered_state["metadata_json"].item()))
    altered_metadata["completed_epoch"] = 0
    altered_state["metadata_json"] = np.asarray(
        json.dumps(altered_metadata, sort_keys=True, separators=(",", ":"))
    )
    np.savez_compressed(state_path, **altered_state)
    with pytest.raises(BehaviorCloningError) as altered:
        run_ppo_pipeline(config_path, resumed, resume=True)
    assert altered.value.code == "ppo_resume_state_hash"

    state_path.write_bytes(b"corrupt resume state")
    with pytest.raises(BehaviorCloningError) as corrupt:
        run_ppo_pipeline(config_path, resumed, resume=True)
    assert corrupt.value.code == "ppo_resume_state_read"


def test_ppo_cli_resume_requires_existing_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path, _, _ = _prepare_config(tmp_path)
    missing_output = tmp_path / "missing-resume"
    exit_code = main(
        [
            "train-ppo",
            "--config",
            str(config_path),
            "--output",
            str(missing_output),
            "--resume",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert json.loads(captured.err)["error"]["code"] == (
        "ppo_resume_output_missing"
    )
    assert not missing_output.exists()


def test_ppo_resume_transaction_reports_stage_failure_and_rolls_back_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, _, _ = _prepare_config(tmp_path)
    output = tmp_path / "resume-output"
    output.mkdir()
    (output / "existing.txt").write_text("original", encoding="utf-8")

    def fail_copytree(*args, **kwargs):
        raise OSError("simulated staging copy failure")

    monkeypatch.setattr(ppo_module.shutil, "copytree", fail_copytree)
    with pytest.raises(BehaviorCloningError) as stage_failure:
        run_ppo_pipeline(config_path, output, resume=True)
    assert stage_failure.value.code == "ppo_resume_stage_write"
    assert (output / "existing.txt").read_text(encoding="utf-8") == "original"
    assert _resume_transaction_residue(tmp_path, output.name) == []

    monkeypatch.undo()
    staging = tmp_path / ".resume-output.ppo-resume-staging-test"
    staging.mkdir()
    (staging / "replacement.txt").write_text("replacement", encoding="utf-8")
    original_replace = ppo_module.os.replace
    replace_calls = 0

    def fail_publish_after_backup(source, destination):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("simulated staging publish failure")
        return original_replace(source, destination)

    monkeypatch.setattr(ppo_module.os, "replace", fail_publish_after_backup)
    with pytest.raises(BehaviorCloningError) as publish_failure:
        ppo_module._publish_resume_staging(staging, output)
    assert publish_failure.value.code == "ppo_resume_publish"
    assert (output / "existing.txt").read_text(encoding="utf-8") == "original"
    assert not staging.exists()
    assert _resume_transaction_residue(tmp_path, output.name) == []
