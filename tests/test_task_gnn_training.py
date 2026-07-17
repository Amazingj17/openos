from __future__ import annotations

import hashlib

import numpy as np
import pytest

import trisched.ppo as ppo_module
from trisched.bc import (
    BehaviorCloningError,
    build_teacher_manifest,
    freeze_task_gnn_teacher_dataset,
    train_task_gnn_bc_baseline,
)
from trisched.env import run_policy, validate_schedule
from trisched.ppo import ValueNetwork, train_task_gnn_ppo
from trisched.scenario import Scenario, generate_dataset


def _source_entries(
    scenarios: list[Scenario],
    split: str,
) -> list[dict[str, object]]:
    return [
        {
            "split": split,
            "split_index": index,
            "source": f"{split}/{scenario.id}.json",
            "source_sha256": hashlib.sha256(
                scenario.id.encode("utf-8")
            ).hexdigest(),
            "scenario_id": scenario.id,
            "scenario_hash": scenario.content_hash(),
        }
        for index, scenario in enumerate(scenarios)
    ]


def _teacher_manifest(
    scenarios: list[Scenario],
    *,
    split: str,
    purpose: str,
) -> dict[str, object]:
    return build_teacher_manifest(
        scenarios,
        _source_entries(scenarios, split),
        split=split,
        purpose=purpose,
        benchmark_manifest_name="micro-task-gnn.json",
        benchmark_manifest_sha256="1" * 64,
        benchmark_id="micro-task-gnn",
        code_metadata={"commit": "unit-test"},
    )


def _micro_data() -> tuple[
    list[Scenario],
    list[Scenario],
    dict[str, object],
    dict[str, object],
]:
    train = generate_dataset(
        4,
        seed=510,
        task_range=(5, 6),
        resource_count=3,
        prefix="task-gnn-train",
    )
    validation = generate_dataset(
        2,
        seed=920,
        task_range=(5, 6),
        resource_count=3,
        prefix="task-gnn-validation",
    )
    return (
        train,
        validation,
        _teacher_manifest(
            train,
            split="train",
            purpose="behavior_cloning_teacher",
        ),
        _teacher_manifest(
            validation,
            split="validation",
            purpose="model_selection_reference",
        ),
    )


def _bc_config() -> dict[str, object]:
    return {
        "hidden_dim": 6,
        "message_dim": 3,
        "epochs": 2,
        "learning_rate": 0.002,
        "gradient_clip": 5.0,
        "shuffle_seed_offset": 17,
        "failure_penalty_ratio": 7.0,
    }


def _ppo_config() -> dict[str, object]:
    return {
        "epochs": 1,
        "episodes_per_epoch": 4,
        "update_epochs": 2,
        "minibatch_size": 8,
        "actor_learning_rate": 0.0003,
        "value_learning_rate": 0.001,
        "value_hidden_dim": 6,
        "gamma": 1.0,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        "entropy_coefficient": 0.001,
        "target_kl": 0.03,
        "gradient_clip": 5.0,
        "shuffle_seed_offset": 29,
        "failure_penalty_ratio": 7.0,
    }


def test_task_gnn_teacher_states_are_frozen_and_test_free() -> None:
    train, validation, train_manifest, validation_manifest = _micro_data()
    train_states = freeze_task_gnn_teacher_dataset(
        train,
        train_manifest,
        split="train",
        purpose="behavior_cloning_teacher",
    )
    validation_states = freeze_task_gnn_teacher_dataset(
        validation,
        validation_manifest,
        split="validation",
        purpose="model_selection_reference",
    )

    assert train_manifest["test_accessed"] is False
    assert validation_manifest["test_accessed"] is False
    assert set(train_states) == {scenario.id for scenario in train}
    assert set(validation_states) == {scenario.id for scenario in validation}
    assert sum(len(states) for states in train_states.values()) == sum(
        scenario.task_count for scenario in train
    )
    for states in (*train_states.values(), *validation_states.values()):
        for state, target in states:
            assert 0 <= target < len(state.actions)
            assert state.features.shape[1] == 14
            assert not state.features.flags.writeable
            assert not state.graph.ranks.flags.writeable
            assert not state.graph.node_features.flags.writeable


def test_task_gnn_bc_and_ppo_micro_training_is_reproducible() -> None:
    train, validation, train_manifest, validation_manifest = _micro_data()
    train_states = freeze_task_gnn_teacher_dataset(
        train,
        train_manifest,
        split="train",
        purpose="behavior_cloning_teacher",
    )
    validation_states = freeze_task_gnn_teacher_dataset(
        validation,
        validation_manifest,
        split="validation",
        purpose="model_selection_reference",
    )
    first_bc = train_task_gnn_bc_baseline(
        train,
        train_manifest,
        validation,
        validation_manifest,
        _bc_config(),
        seed=61,
        train_frozen_states=train_states,
        validation_frozen_states=validation_states,
    )
    second_bc = train_task_gnn_bc_baseline(
        train,
        train_manifest,
        validation,
        validation_manifest,
        _bc_config(),
        seed=61,
        train_frozen_states=train_states,
        validation_frozen_states=validation_states,
    )

    first_best_bc, first_last_bc, first_bc_summary = first_bc
    second_best_bc, second_last_bc, second_bc_summary = second_bc
    assert first_bc_summary == second_bc_summary
    assert first_bc_summary["selection"]["test_accessed"] is False
    assert len(first_bc_summary["epochs"]) == 2
    assert all(
        np.array_equal(first_best_bc.params[name], second_best_bc.params[name])
        and np.array_equal(first_last_bc.params[name], second_last_bc.params[name])
        for name in first_best_bc.params
    )
    assert all(
        record["train_illegal_action_count"] == 0
        and record["validation"]["failure_count"] == 0
        for record in first_bc_summary["epochs"]
    )

    first_ppo = train_task_gnn_ppo(
        first_best_bc,
        train,
        train_manifest,
        validation,
        validation_manifest,
        _ppo_config(),
        seed=61,
        validation_frozen_states=validation_states,
    )
    second_ppo = train_task_gnn_ppo(
        second_best_bc,
        train,
        train_manifest,
        validation,
        validation_manifest,
        _ppo_config(),
        seed=61,
        validation_frozen_states=validation_states,
    )
    first_best, first_last, _, _, first_summary = first_ppo
    second_best, second_last, _, _, second_summary = second_ppo

    assert first_summary == second_summary
    assert first_summary["selection"]["test_accessed"] is False
    assert first_summary["selection"]["best_epoch"] in {0, 1}
    assert first_summary["epochs"][0]["source"] == (
        "task_gnn_behavior_cloning_warm_start"
    )
    ppo_epoch = first_summary["epochs"][1]
    assert ppo_epoch["source"] == "task_gnn_masked_ppo"
    assert ppo_epoch["reward_identity_max_abs_error"] <= 1e-9
    assert ppo_epoch["validation"]["failure_count"] == 0
    assert ppo_epoch["validation"]["illegal_action_count"] == 0
    assert ppo_epoch["update"]["transition_count"] > 0
    assert all(
        np.array_equal(first_best.params[name], second_best.params[name])
        and np.array_equal(first_last.params[name], second_last.params[name])
        for name in first_best.params
    )
    assert any(
        not np.array_equal(first_last.params[name], first_best_bc.params[name])
        for name in first_last.params
    )
    for policy in (first_best_bc, first_best, first_last):
        result = run_policy(validation[0], policy)
        validate_schedule(validation[0], result)


def test_task_gnn_ppo_transition_contains_only_frozen_replay_state() -> None:
    train, validation, train_manifest, validation_manifest = _micro_data()
    train_states = freeze_task_gnn_teacher_dataset(
        train,
        train_manifest,
        split="train",
        purpose="behavior_cloning_teacher",
    )
    warm_start, _, _ = train_task_gnn_bc_baseline(
        train,
        train_manifest,
        validation,
        validation_manifest,
        _bc_config(),
        seed=67,
        train_frozen_states=train_states,
    )
    record = next(
        entry
        for entry in train_manifest["entries"]
        if entry["scenario_id"] == train[0].id
    )
    critic = ValueNetwork(feature_dim=14, hidden_dim=4, seed=71)
    transitions, ratio, reward_error = ppo_module._collect_task_gnn_episode(
        warm_start,
        critic,
        train[0],
        float(record["makespan"]),
        gamma=1.0,
        gae_lambda=0.95,
    )

    assert len(transitions) == train[0].task_count
    assert ratio > 0
    assert reward_error <= 1e-9
    assert set(vars(transitions[0])) == {
        "frozen_state",
        "selected_index",
        "old_log_probability",
        "state_features",
        "advantage",
        "return_value",
    }
    assert not transitions[0].frozen_state.features.flags.writeable
    assert not transitions[0].state_features.flags.writeable
    assert all(
        transition.frozen_state.graph is transitions[0].frozen_state.graph
        for transition in transitions
    )


def test_task_gnn_ppo_rejects_reward_contract_drift() -> None:
    train, validation, train_manifest, validation_manifest = _micro_data()
    warm_start, _, _ = train_task_gnn_bc_baseline(
        train,
        train_manifest,
        validation,
        validation_manifest,
        _bc_config(),
        seed=73,
    )
    config = _ppo_config()
    config["gamma"] = 0.99
    with pytest.raises(BehaviorCloningError) as captured:
        train_task_gnn_ppo(
            warm_start,
            train,
            train_manifest,
            validation,
            validation_manifest,
            config,
            seed=73,
        )
    assert captured.value.code == "ppo_reward_contract"
    assert captured.value.path == "$.ppo.gamma"
