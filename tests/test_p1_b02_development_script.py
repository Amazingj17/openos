from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_p1_b02_development import build_runner_bundle
from trisched.env import validate_schedule
from trisched.gnn import TASK_GNN_FEATURE_NAMES, TaskGNNPolicy
from trisched.learning import (
    FEATURE_NAMES,
    TEACHER_FEATURE_NAMES,
    MaskedMLPPolicy,
)
from trisched.scenario import generate_scenario


def _contract() -> dict:
    return {
        "policies": [
            {"id": "heft", "required_seeds": [0]},
            {"id": "greedy_eft", "required_seeds": [0]},
            {"id": "cpop", "required_seeds": [0]},
            {"id": "random", "required_seeds": [7]},
            {"id": "bc", "required_seeds": [31]},
            {"id": "masked_mlp", "required_seeds": [31]},
            {"id": "task_gnn", "required_seeds": [31]},
        ]
    }


def _checkpoints(tmp_path: Path) -> tuple[Path, Path, Path]:
    bc = tmp_path / "bc" / "bc_best.npz"
    masked = tmp_path / "masked"
    task_gnn = tmp_path / "task-gnn"
    MaskedMLPPolicy(
        hidden_dim=8,
        seed=31,
        feature_names=FEATURE_NAMES,
    ).save(bc)
    MaskedMLPPolicy(
        hidden_dim=8,
        seed=31,
        feature_names=tuple(
            name for name in FEATURE_NAMES if name not in TEACHER_FEATURE_NAMES
        ),
    ).save(masked / "seed_31_ppo_best_policy.npz")
    TaskGNNPolicy(
        hidden_dim=8,
        message_dim=4,
        seed=31,
        feature_names=TASK_GNN_FEATURE_NAMES,
    ).save(task_gnn / "seed_31_task_gnn_ppo_best_policy.npz")
    return bc, masked, task_gnn


def test_frozen_development_runner_bundle_loads_exact_grid_and_runs(
    tmp_path: Path,
) -> None:
    bc, masked, task_gnn = _checkpoints(tmp_path)
    runners, metadata = build_runner_bundle(
        _contract(),
        repository=tmp_path,
        bc_checkpoint=bc,
        masked_mlp_dir=masked,
        task_gnn_dir=task_gnn,
    )
    expected = {
        ("heft", 0),
        ("greedy_eft", 0),
        ("cpop", 0),
        ("random", 7),
        ("bc", 31),
        ("masked_mlp", 31),
        ("task_gnn", 31),
    }
    assert set(runners) == expected
    assert set(metadata["checkpoints"]) == {"bc", "masked_mlp", "task_gnn"}
    assert metadata["checkpoints"]["bc"]["31"]["internal_seed"] == 31
    assert metadata["checkpoints"]["masked_mlp"]["31"]["feature_names"] == [
        name for name in FEATURE_NAMES if name not in TEACHER_FEATURE_NAMES
    ]
    assert metadata["checkpoints"]["task_gnn"]["31"]["feature_names"] == list(
        TASK_GNN_FEATURE_NAMES
    )
    scenario = generate_scenario(seed=123, task_count=6)
    for policy, seed in expected:
        result = runners[(policy, seed)].schedule(scenario)
        assert result.policy_name == policy
        validate_schedule(scenario, result)


def test_frozen_development_runner_bundle_rejects_checkpoint_seed(
    tmp_path: Path,
) -> None:
    bc, masked, task_gnn = _checkpoints(tmp_path)
    MaskedMLPPolicy(
        hidden_dim=8,
        seed=99,
        feature_names=tuple(
            name for name in FEATURE_NAMES if name not in TEACHER_FEATURE_NAMES
        ),
    ).save(masked / "seed_31_ppo_best_policy.npz")
    with pytest.raises(ValueError, match="checkpoint seed mismatch"):
        build_runner_bundle(
            _contract(),
            repository=tmp_path,
            bc_checkpoint=bc,
            masked_mlp_dir=masked,
            task_gnn_dir=task_gnn,
        )
