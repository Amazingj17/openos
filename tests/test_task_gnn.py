from __future__ import annotations

import numpy as np
import pytest

from trisched.env import HeterogeneousDagEnv, run_policy, validate_schedule
from trisched.gnn import (
    TASK_GNN_FEATURE_NAMES,
    TASK_NODE_FEATURE_NAMES,
    TaskGNNPolicy,
    task_gnn_metadata,
)
from trisched.learning import FEATURE_NAMES, MaskedMLPPolicy
from trisched.scenario import Edge, Resource, Scenario, Task


def _scenario(*, alternative_graph: bool = False) -> Scenario:
    edges = (
        (Edge(0, 2, 2.0), Edge(1, 2, 1.0), Edge(2, 3, 1.0))
        if not alternative_graph
        else (Edge(0, 2, 2.0), Edge(1, 3, 1.0), Edge(2, 3, 1.0))
    )
    return Scenario(
        id="task-gnn-alternative" if alternative_graph else "task-gnn",
        seed=7,
        tasks=(
            Task(0, 6.0),
            Task(1, 4.0),
            Task(2, 8.0),
            Task(3, 3.0),
        ),
        resources=(
            Resource(0, "device-0", "device", 1.0),
            Resource(1, "cloud-0", "cloud", 3.0),
        ),
        edges=edges,
        bandwidth=((1e9, 4.0), (4.0, 1e9)),
        latency=((0.0, 0.5), (0.5, 0.0)),
    )


def test_task_gnn_preserves_14d_input_and_legal_action_mask() -> None:
    scenario = _scenario()
    env = HeterogeneousDagEnv(scenario)
    policy = TaskGNNPolicy(hidden_dim=8, message_dim=4, seed=11)
    cache = policy.distribution(env)

    assert len(TASK_GNN_FEATURE_NAMES) == 14
    assert set(TASK_GNN_FEATURE_NAMES) == set(FEATURE_NAMES) - {
        "is_heft_task",
        "is_heft_pair",
    }
    assert cache.actions == env.candidate_actions()
    assert cache.features.shape == (len(cache.actions), 14)
    assert cache.node_features.shape == (
        scenario.task_count,
        len(TASK_NODE_FEATURE_NAMES),
    )
    assert cache.node_context.shape == (scenario.task_count, 4)
    assert np.isfinite(cache.probabilities).all()
    assert np.all(cache.probabilities > 0)
    assert np.sum(cache.probabilities) == pytest.approx(1.0)
    assert policy.select_action(env) in env.candidate_actions()

    result = run_policy(scenario, policy)
    validate_schedule(scenario, result)
    assert result.policy_name == "task_gnn"


def test_task_gnn_is_seed_deterministic_and_graph_sensitive() -> None:
    scenario = _scenario()
    alternative = _scenario(alternative_graph=True)
    first = TaskGNNPolicy(hidden_dim=8, message_dim=4, seed=23)
    second = TaskGNNPolicy(hidden_dim=8, message_dim=4, seed=23)

    first_cache = first.distribution(HeterogeneousDagEnv(scenario))
    second_cache = second.distribution(HeterogeneousDagEnv(scenario))
    alternative_cache = first.distribution(HeterogeneousDagEnv(alternative))

    assert all(
        np.array_equal(first.params[name], second.params[name])
        for name in first.params
    )
    assert np.array_equal(
        first_cache.probabilities,
        second_cache.probabilities,
    )
    assert not np.allclose(
        first_cache.node_context,
        alternative_cache.node_context,
    )


def test_task_gnn_checkpoint_and_parameter_metadata_round_trip(tmp_path) -> None:
    scenario = _scenario()
    policy = TaskGNNPolicy(hidden_dim=8, message_dim=4, seed=31)
    before = policy.distribution(HeterogeneousDagEnv(scenario))
    path = tmp_path / "task_gnn.npz"
    policy.save(path)
    loaded = TaskGNNPolicy.load(path)
    after = loaded.distribution(HeterogeneousDagEnv(scenario))

    assert np.array_equal(before.probabilities, after.probabilities)
    assert all(
        np.array_equal(policy.params[name], loaded.params[name])
        for name in policy.params
    )
    assert loaded.parameter_count == 232
    mlp = MaskedMLPPolicy(
        hidden_dim=8,
        seed=31,
        feature_names=TASK_GNN_FEATURE_NAMES,
    )
    assert loaded.parameter_count > sum(
        int(value.size) for value in mlp.params.values()
    )
    assert task_gnn_metadata(loaded) == {
        "architecture": "task_gnn_v1",
        "base_feature_count": 14,
        "base_feature_names": list(TASK_GNN_FEATURE_NAMES),
        "node_feature_names": list(TASK_NODE_FEATURE_NAMES),
        "message_passing_steps": 1,
        "message_directions": ["predecessor", "successor"],
        "hidden_dim": 8,
        "message_dim": 4,
        "parameter_count": 232,
    }


def test_task_gnn_rejects_feature_contract_drift() -> None:
    with pytest.raises(ValueError, match="canonical 14-D"):
        TaskGNNPolicy(feature_names=FEATURE_NAMES)
    with pytest.raises(ValueError, match="canonical 14-D"):
        TaskGNNPolicy(feature_names=TASK_GNN_FEATURE_NAMES[:-1])
