from __future__ import annotations

import json
import numpy as np

from trisched.dual_graph import (
    DUAL_GRAPH_PAIR_FEATURE_NAMES,
    RESOURCE_NODE_FEATURE_NAMES,
    TASK_NODE_FEATURE_NAMES,
    CloudEdgeDualGraphPolicy,
    freeze_dual_graph,
    freeze_dual_graph_state,
    run_complex_dual_graph_pipeline,
    train_dual_graph_ppo,
)
from trisched.env import HeterogeneousDagEnv, run_policy
from trisched.scenario import generate_complex_scenario


def test_dual_graph_encodes_task_data_and_resource_network() -> None:
    scenario = generate_complex_scenario(10, task_count=10, resource_count=6)
    graph = freeze_dual_graph(scenario)
    assert graph.task_features.shape == (10, len(TASK_NODE_FEATURE_NAMES))
    assert graph.resource_static_features.shape == (6, len(RESOURCE_NODE_FEATURE_NAMES) - 2)
    assert graph.resource_adjacency.shape == (6, 6)
    assert not graph.task_features.flags.writeable
    assert not graph.resource_adjacency.flags.writeable
    assert np.allclose(np.diag(graph.resource_adjacency), 0.0)


def test_actor_scores_only_masked_legal_task_resource_pairs() -> None:
    scenario = generate_complex_scenario(11, task_count=12, resource_count=6)
    env = HeterogeneousDagEnv(scenario)
    state = freeze_dual_graph_state(env)
    policy = CloudEdgeDualGraphPolicy(seed=5)
    cache = policy.distribution_from_frozen_state(state)
    assert cache.actions == env.candidate_actions()
    assert cache.pair_features.shape == (
        len(cache.actions),
        len(DUAL_GRAPH_PAIR_FEATURE_NAMES),
    )
    assert np.isclose(np.sum(cache.probabilities), 1.0)
    assert np.all(cache.probabilities > 0)
    assert run_policy(scenario, policy).makespan > 0


def test_dual_graph_analytic_gradient_matches_finite_difference() -> None:
    scenario = generate_complex_scenario(12, task_count=7, resource_count=5)
    state = freeze_dual_graph_state(HeterogeneousDagEnv(scenario))
    policy = CloudEdgeDualGraphPolicy(
        hidden_dim=6, task_message_dim=4, resource_message_dim=3, seed=2
    )
    cache = policy.distribution_from_frozen_state(state)
    selected = min(1, len(cache.actions) - 1)
    analytic = policy.log_probability_gradients(cache, selected)
    epsilon = 1e-6
    checks = (
        ("task_node_w", (0, 0)),
        ("resource_neighbor_w", (0, 0)),
        ("pair_w", (0, 0)),
        ("output_w", (0,)),
    )
    for name, index in checks:
        original = float(policy.params[name][index])
        policy.params[name][index] = original + epsilon
        positive = np.log(
            policy.distribution_from_frozen_state(state).probabilities[selected]
        )
        policy.params[name][index] = original - epsilon
        negative = np.log(
            policy.distribution_from_frozen_state(state).probabilities[selected]
        )
        policy.params[name][index] = original
        numerical = (positive - negative) / (2.0 * epsilon)
        assert np.isclose(analytic[name][index], numerical, atol=2e-5, rtol=2e-4)


def test_dual_graph_checkpoint_and_ppo_smoke(tmp_path) -> None:
    scenarios = [
        generate_complex_scenario(seed, task_count=8, resource_count=5)
        for seed in (20, 21)
    ]
    actor, critic, history = train_dual_graph_ppo(
        scenarios,
        epochs=1,
        episodes_per_epoch=2,
        hidden_dim=8,
        task_message_dim=4,
        resource_message_dim=4,
        value_hidden_dim=8,
        imitation_epochs=1,
        update_epochs=1,
        minibatch_size=8,
        seed=3,
    )
    assert critic.feature_dim == len(DUAL_GRAPH_PAIR_FEATURE_NAMES)
    assert history[0]["transition_count"] == 16
    checkpoint = tmp_path / "dual_graph.npz"
    actor.save(checkpoint)
    restored = CloudEdgeDualGraphPolicy.load(checkpoint)
    original_result = run_policy(scenarios[0], actor)
    restored_result = run_policy(scenarios[0], restored)
    assert restored_result.to_dict() == original_result.to_dict()


def test_one_command_pipeline_writes_standard_outputs(tmp_path) -> None:
    output = tmp_path / "output"
    config = {
        "seed": 9,
        "output_dir": str(output),
        "dataset": {
            "train_count": 2,
            "validation_count": 2,
            "task_range": [5, 6],
            "resource_count": 3,
            "edge_probability": 0.1,
        },
        "training": {
            "epochs": 1,
            "episodes_per_epoch": 2,
            "hidden_dim": 8,
            "task_message_dim": 4,
            "resource_message_dim": 4,
            "value_hidden_dim": 8,
            "imitation_epochs": 1,
            "imitation_learning_rate": 0.003,
            "actor_learning_rate": 0.0003,
            "value_learning_rate": 0.001,
            "gae_lambda": 0.95,
            "clip_ratio": 0.2,
            "entropy_coefficient": 0.01,
            "update_epochs": 1,
            "minibatch_size": 8,
            "gradient_clip": 0.5,
        },
    }
    source = tmp_path / "config.json"
    source.write_text(json.dumps(config), encoding="utf-8")
    summary_path = run_complex_dual_graph_pipeline(source)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["primary_metric"]["sample_count"] == 2
    assert summary["competition_scope"]["resource_kinds"] == [
        "device",
        "edge",
        "cloud",
    ]
    assert (output / "actor.npz").is_file()
    assert (output / "critic.npz").is_file()
    assert (output / "validation_per_instance.csv").is_file()
