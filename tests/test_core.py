from __future__ import annotations

import json

import numpy as np
import pytest

from trisched.cli import run_pipeline
from trisched.env import HeterogeneousDagEnv, run_policy, validate_schedule
from trisched.learning import MaskedMLPPolicy, candidate_features, train_policy
from trisched.policies import HeftPolicy, RandomPolicy, compute_upward_ranks
from trisched.scenario import Edge, Resource, Scenario, Task, generate_dataset


def toy_scenario() -> Scenario:
    return Scenario(
        id="toy",
        seed=7,
        tasks=(Task(0, 6.0), Task(1, 4.0), Task(2, 8.0), Task(3, 3.0)),
        resources=(
            Resource(0, "device-0", "device", 1.0),
            Resource(1, "cloud-0", "cloud", 3.0),
        ),
        edges=(Edge(0, 2, 2.0), Edge(1, 2, 1.0), Edge(2, 3, 1.0)),
        bandwidth=((1e9, 4.0), (4.0, 1e9)),
        latency=((0.0, 0.5), (0.5, 0.0)),
    )


def test_cycle_is_rejected() -> None:
    with pytest.raises(ValueError, match="acyclic"):
        Scenario(
            id="cycle",
            seed=1,
            tasks=(Task(0, 1.0), Task(1, 1.0)),
            resources=(Resource(0, "cloud-0", "cloud", 1.0),),
            edges=(Edge(0, 1, 0.0), Edge(1, 0, 0.0)),
            bandwidth=((1e9,),),
            latency=((0.0,),),
        )


def test_heft_produces_a_valid_complete_schedule() -> None:
    scenario = toy_scenario()
    result = run_policy(scenario, HeftPolicy())
    validate_schedule(scenario, result)
    assert len(result.entries) == scenario.task_count
    assert result.makespan > 0


def test_ready_mask_only_exposes_legal_pairs() -> None:
    scenario = toy_scenario()
    env = HeterogeneousDagEnv(scenario)
    assert env.ready_tasks() == (0, 1)
    assert set(env.candidate_actions()) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    with pytest.raises(ValueError, match="not ready"):
        env.step(2, 0)
    env.step(0, 0)
    assert env.ready_tasks() == (1,)


def test_feature_shape_matches_all_masked_actions() -> None:
    scenario = toy_scenario()
    env = HeterogeneousDagEnv(scenario)
    actions, features = candidate_features(env, compute_upward_ranks(scenario))
    assert len(actions) == len(env.ready_tasks()) * scenario.resource_count
    assert features.shape == (len(actions), 16)
    assert np.isfinite(features).all()


def test_minimal_training_checkpoint_round_trip(tmp_path) -> None:
    scenarios = generate_dataset(4, seed=100, task_range=(5, 7), prefix="train")
    policy, history = train_policy(
        scenarios,
        {
            "hidden_dim": 8,
            "imitation_epochs": 1,
            "reinforce_epochs": 1,
            "imitation_learning_rate": 0.005,
            "reinforce_learning_rate": 0.001,
            "reinforce_temperature": 2.0,
            "gradient_clip": 5.0,
        },
        seed=11,
    )
    path = tmp_path / "model.npz"
    policy.save(path)
    loaded = MaskedMLPPolicy.load(path)
    result = run_policy(scenarios[0], loaded)
    validate_schedule(scenarios[0], result)
    assert history["imitation"]
    assert history["reinforce"]


def test_pipeline_writes_standard_outputs(tmp_path) -> None:
    output = tmp_path / "run"
    config = {
        "seed": 42,
        "output_dir": str(output),
        "dataset": {
            "train_count": 4,
            "validation_count": 3,
            "test_count": 3,
            "task_range": [5, 7],
            "resource_count": 3,
            "edge_probability": 0.2,
        },
        "training": {
            "hidden_dim": 8,
            "imitation_epochs": 1,
            "reinforce_epochs": 1,
            "imitation_learning_rate": 0.005,
            "reinforce_learning_rate": 0.001,
            "reinforce_temperature": 2.0,
            "gradient_clip": 5.0,
        },
        "evaluation": {"random_seed": 9},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    summary_path = run_pipeline(config_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["primary_metric"]["name"] == "mean_ratio"
    assert summary["test"]["heft"]["mean_ratio"] == pytest.approx(1.0)
    assert (output / "masked_mlp.npz").exists()
    assert (output / "test_per_instance.csv").exists()
    assert (output / "dataset_manifest.json").exists()
