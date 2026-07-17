from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from trisched.bc import BehaviorCloningError
from trisched.benchmark import build_stg_manifest, load_frozen_split
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
    assert run_manifest["inputs"]["test_accessed"] is False
    assert set(run_manifest["inputs"]["splits"]) == {"train", "validation"}
    for name, metadata in run_manifest["artifacts"].items():
        artifact = first / name
        assert artifact.stat().st_size == metadata["bytes"]
        assert _file_hash(artifact) == metadata["sha256"]
