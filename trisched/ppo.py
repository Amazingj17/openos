from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .bc import (
    BehaviorCloningError,
    FrozenTaskGNNTeacherStates,
    FrozenTeacherStates,
    _file_hash,
    _git_metadata,
    _resolve_config_path,
    _write_json,
    _write_jsonl,
    build_teacher_manifest,
    evaluate_bc_policy,
    freeze_task_gnn_teacher_dataset,
    freeze_teacher_dataset,
    policy_parameter_hash,
    train_task_gnn_bc_baseline,
    train_bc_baseline,
)
from .benchmark import load_benchmark_manifest, load_frozen_split
from .env import HeterogeneousDagEnv, validate_schedule
from .gnn import (
    TASK_GNN_FEATURE_NAMES,
    FrozenTaskGNNState,
    TaskGNNPolicy,
    freeze_task_gnn_state,
    freeze_task_graph,
    task_gnn_metadata,
    task_gnn_parameter_hash,
)
from .learning import (
    FEATURE_NAMES,
    TEACHER_FEATURE_NAMES,
    MaskedMLPPolicy,
)
from .oracle import validate_schedule_independent
from .scenario import Scenario


def _fail(
    code: str,
    path: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> None:
    raise BehaviorCloningError(
        code,
        path,
        message,
        details=details,
    )


def _positive_integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail("config_value", path, "expected a positive integer")
    return value


def _finite_number(
    value: Any,
    path: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("config_value", path, "expected a number")
    result = float(value)
    if not math.isfinite(result):
        _fail("config_value", path, "expected a finite number")
    if positive and result <= 0:
        _fail("config_value", path, "expected a positive number")
    if non_negative and result < 0:
        _fail("config_value", path, "expected a non-negative number")
    return result


def load_ppo_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail("config_read", "$", str(error))
    if not isinstance(payload, dict):
        _fail("config_type", "$", "expected an object")
    if payload.get("format_version") != 1:
        _fail("config_version", "$.format_version", "expected 1")

    seeds = payload.get("seeds")
    if not isinstance(seeds, list) or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds
    ):
        _fail("config_value", "$.seeds", "expected an array of integers")
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        _fail(
            "ppo_seed_contract",
            "$.seeds",
            "P1-A02 requires at least three distinct training seeds",
        )

    output_dir = payload.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir.strip():
        _fail("config_value", "$.output_dir", "expected a non-empty path")
    benchmark = payload.get("benchmark")
    if not isinstance(benchmark, dict):
        _fail("config_type", "$.benchmark", "expected an object")
    for key in ("manifest", "raw_root"):
        if not isinstance(benchmark.get(key), str) or not benchmark[key].strip():
            _fail(
                "config_value",
                f"$.benchmark.{key}",
                "expected a non-empty path",
            )

    feature_config = payload.get("features")
    if not isinstance(feature_config, dict):
        _fail("config_type", "$.features", "expected an object")
    excluded = feature_config.get("exclude")
    if not isinstance(excluded, list) or any(
        not isinstance(name, str) for name in excluded
    ):
        _fail("config_value", "$.features.exclude", "expected feature names")
    if len(excluded) != len(set(excluded)):
        _fail("config_value", "$.features.exclude", "features must be unique")
    unknown = sorted(set(excluded) - set(FEATURE_NAMES))
    if unknown:
        _fail(
            "config_value",
            "$.features.exclude",
            "unknown feature names",
            details={"unknown": unknown},
        )
    missing_teacher_exclusions = sorted(
        set(TEACHER_FEATURE_NAMES) - set(excluded)
    )
    if missing_teacher_exclusions:
        _fail(
            "ppo_teacher_feature_leakage",
            "$.features.exclude",
            "masked PPO must exclude direct HEFT decision features",
            details={"missing": missing_teacher_exclusions},
        )
    selected_features = tuple(
        name for name in FEATURE_NAMES if name not in set(excluded)
    )
    if not selected_features:
        _fail("config_value", "$.features.exclude", "no features remain")

    bc = payload.get("behavior_cloning")
    if not isinstance(bc, dict):
        _fail("config_type", "$.behavior_cloning", "expected an object")
    normalized_bc = {
        "hidden_dim": _positive_integer(
            bc.get("hidden_dim", 32), "$.behavior_cloning.hidden_dim"
        ),
        "epochs": _positive_integer(
            bc.get("epochs", 1), "$.behavior_cloning.epochs"
        ),
        "learning_rate": _finite_number(
            bc.get("learning_rate", 0.004),
            "$.behavior_cloning.learning_rate",
            positive=True,
        ),
        "gradient_clip": _finite_number(
            bc.get("gradient_clip", 5.0),
            "$.behavior_cloning.gradient_clip",
            non_negative=True,
        ),
        "shuffle_seed_offset": _positive_integer(
            bc.get("shuffle_seed_offset", 101),
            "$.behavior_cloning.shuffle_seed_offset",
        ),
    }

    ppo = payload.get("ppo")
    if not isinstance(ppo, dict):
        _fail("config_type", "$.ppo", "expected an object")
    gamma = _finite_number(ppo.get("gamma", 1.0), "$.ppo.gamma", positive=True)
    if abs(gamma - 1.0) > 1e-12:
        _fail(
            "ppo_reward_contract",
            "$.ppo.gamma",
            "gamma must be 1.0 so shaped rewards sum to negative final ratio",
        )
    gae_lambda = _finite_number(
        ppo.get("gae_lambda", 0.95),
        "$.ppo.gae_lambda",
        non_negative=True,
    )
    if gae_lambda > 1.0:
        _fail("config_value", "$.ppo.gae_lambda", "expected a value in [0, 1]")
    clip_ratio = _finite_number(
        ppo.get("clip_ratio", 0.2), "$.ppo.clip_ratio", positive=True
    )
    if clip_ratio >= 1.0:
        _fail("config_value", "$.ppo.clip_ratio", "expected a value below 1")
    normalized_ppo = {
        "epochs": _positive_integer(ppo.get("epochs", 2), "$.ppo.epochs"),
        "episodes_per_epoch": _positive_integer(
            ppo.get("episodes_per_epoch", 120),
            "$.ppo.episodes_per_epoch",
        ),
        "update_epochs": _positive_integer(
            ppo.get("update_epochs", 2), "$.ppo.update_epochs"
        ),
        "minibatch_size": _positive_integer(
            ppo.get("minibatch_size", 256), "$.ppo.minibatch_size"
        ),
        "actor_learning_rate": _finite_number(
            ppo.get("actor_learning_rate", 0.0003),
            "$.ppo.actor_learning_rate",
            positive=True,
        ),
        "value_learning_rate": _finite_number(
            ppo.get("value_learning_rate", 0.001),
            "$.ppo.value_learning_rate",
            positive=True,
        ),
        "value_hidden_dim": _positive_integer(
            ppo.get("value_hidden_dim", 32), "$.ppo.value_hidden_dim"
        ),
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_ratio": clip_ratio,
        "entropy_coefficient": _finite_number(
            ppo.get("entropy_coefficient", 0.001),
            "$.ppo.entropy_coefficient",
            non_negative=True,
        ),
        "target_kl": _finite_number(
            ppo.get("target_kl", 0.03), "$.ppo.target_kl", positive=True
        ),
        "gradient_clip": _finite_number(
            ppo.get("gradient_clip", 5.0),
            "$.ppo.gradient_clip",
            non_negative=True,
        ),
        "shuffle_seed_offset": _positive_integer(
            ppo.get("shuffle_seed_offset", 303),
            "$.ppo.shuffle_seed_offset",
        ),
    }

    selection = payload.get("selection", {})
    if not isinstance(selection, dict):
        _fail("config_type", "$.selection", "expected an object")
    metric = selection.get("metric", "validation_mean_ratio")
    if metric != "validation_mean_ratio":
        _fail(
            "selection_metric",
            "$.selection.metric",
            "only validation_mean_ratio is supported",
        )
    failure_penalty = _finite_number(
        selection.get("failure_penalty_ratio", 10.0),
        "$.selection.failure_penalty_ratio",
        positive=True,
    )
    target_ratio = _finite_number(
        selection.get("target_ratio", 1.0),
        "$.selection.target_ratio",
        positive=True,
    )
    ablation = payload.get("ablation", {})
    if not isinstance(ablation, dict):
        _fail("config_type", "$.ablation", "expected an object")
    reference_seed = ablation.get("teacher_feature_reference_seed", seeds[0])
    if reference_seed not in seeds:
        _fail(
            "config_value",
            "$.ablation.teacher_feature_reference_seed",
            "reference seed must be one of the training seeds",
        )

    normalized_extension: dict[str, Any] | None = None
    seed_extension = payload.get("seed_extension")
    if seed_extension is not None:
        if not isinstance(seed_extension, dict):
            _fail("config_type", "$.seed_extension", "expected an object")
        source_dir = seed_extension.get("source_dir")
        if not isinstance(source_dir, str) or not source_dir.strip():
            _fail(
                "config_value",
                "$.seed_extension.source_dir",
                "expected a non-empty path",
            )
        source_manifest_sha256 = seed_extension.get("run_manifest_sha256")
        if (
            not isinstance(source_manifest_sha256, str)
            or len(source_manifest_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in source_manifest_sha256
            )
        ):
            _fail(
                "config_value",
                "$.seed_extension.run_manifest_sha256",
                "expected a lowercase SHA-256",
            )
        reuse_seeds = seed_extension.get("reuse_seeds")
        if (
            not isinstance(reuse_seeds, list)
            or not reuse_seeds
            or any(
                isinstance(seed, bool) or not isinstance(seed, int)
                for seed in reuse_seeds
            )
            or len(set(reuse_seeds)) != len(reuse_seeds)
        ):
            _fail(
                "config_value",
                "$.seed_extension.reuse_seeds",
                "expected distinct integer seeds",
            )
        if seeds[: len(reuse_seeds)] != reuse_seeds:
            _fail(
                "ppo_seed_extension",
                "$.seed_extension.reuse_seeds",
                "reused seeds must be an exact prefix of target seeds",
            )
        if len(reuse_seeds) == len(seeds):
            _fail(
                "ppo_seed_extension",
                "$.seeds",
                "seed extension must add at least one new seed",
            )
        normalized_extension = {
            "source_dir": source_dir,
            "run_manifest_sha256": source_manifest_sha256,
            "reuse_seeds": list(reuse_seeds),
        }

    normalized = {
        "format_version": 1,
        "seeds": list(seeds),
        "output_dir": output_dir,
        "benchmark": {
            "manifest": benchmark["manifest"],
            "raw_root": benchmark["raw_root"],
        },
        "features": {
            "all": list(FEATURE_NAMES),
            "excluded": list(excluded),
            "selected": list(selected_features),
        },
        "behavior_cloning": normalized_bc,
        "ppo": normalized_ppo,
        "selection": {
            "metric": metric,
            "failure_penalty_ratio": failure_penalty,
            "target_ratio": target_ratio,
            "tie_break": ["zero_failures", "lower_mean_ratio", "earlier_epoch"],
        },
        "ablation": {
            "teacher_feature_reference_seed": reference_seed,
        },
    }
    if normalized_extension is not None:
        normalized["seed_extension"] = normalized_extension
    return normalized


def load_task_gnn_config(path: str | Path) -> dict[str, Any]:
    """Load the P1-A03 task-GNN contract without changing masked-MLP PPO."""

    source = Path(path)
    base = load_ppo_config(source)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail("config_read", "$", str(error))
    allowed_keys = {
        "$": {
            "format_version",
            "seeds",
            "output_dir",
            "benchmark",
            "features",
            "task_gnn",
            "behavior_cloning",
            "ppo",
            "selection",
        },
        "$.benchmark": {"manifest", "raw_root"},
        "$.features": {"exclude"},
        "$.task_gnn": {"architecture", "message_dim"},
        "$.behavior_cloning": {
            "hidden_dim",
            "epochs",
            "learning_rate",
            "gradient_clip",
            "shuffle_seed_offset",
        },
        "$.ppo": {
            "epochs",
            "episodes_per_epoch",
            "update_epochs",
            "minibatch_size",
            "actor_learning_rate",
            "value_learning_rate",
            "value_hidden_dim",
            "gamma",
            "gae_lambda",
            "clip_ratio",
            "entropy_coefficient",
            "target_kl",
            "gradient_clip",
            "shuffle_seed_offset",
        },
        "$.selection": {
            "metric",
            "failure_penalty_ratio",
            "target_ratio",
        },
    }
    config_objects = {
        "$": payload,
        "$.benchmark": payload.get("benchmark"),
        "$.features": payload.get("features"),
        "$.task_gnn": payload.get("task_gnn"),
        "$.behavior_cloning": payload.get("behavior_cloning"),
        "$.ppo": payload.get("ppo"),
        "$.selection": payload.get("selection"),
    }
    for object_path, allowed in allowed_keys.items():
        value = config_objects[object_path]
        if not isinstance(value, dict):
            continue
        unknown_keys = sorted(set(value) - allowed)
        if unknown_keys:
            _fail(
                "task_gnn_config_variable",
                object_path,
                "task-GNN v1 rejects unreviewed experiment variables",
                details={"unknown": unknown_keys},
            )
    task_gnn = payload.get("task_gnn")
    if not isinstance(task_gnn, dict):
        _fail("config_type", "$.task_gnn", "expected an object")
    architecture = task_gnn.get("architecture")
    if architecture != TaskGNNPolicy.architecture:
        _fail(
            "task_gnn_architecture",
            "$.task_gnn.architecture",
            f"expected {TaskGNNPolicy.architecture!r}",
        )
    if tuple(base["features"]["selected"]) != TASK_GNN_FEATURE_NAMES:
        _fail(
            "task_gnn_feature_schema",
            "$.features.exclude",
            "task-GNN v1 requires the canonical 14-D teacher-free schema",
            details={
                "expected": list(TASK_GNN_FEATURE_NAMES),
                "actual": base["features"]["selected"],
            },
        )
    normalized = dict(base)
    normalized.pop("ablation", None)
    normalized["task_gnn"] = {
        "architecture": architecture,
        "message_dim": _positive_integer(
            task_gnn.get("message_dim", 8),
            "$.task_gnn.message_dim",
        ),
    }
    return normalized


class ValueNetwork:
    """Small state-value MLP over mean/max pooled legal-candidate features."""

    def __init__(self, feature_dim: int, hidden_dim: int, seed: int) -> None:
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("value network dimensions must be positive")
        self.feature_dim = feature_dim
        self.input_dim = feature_dim * 2
        self.hidden_dim = hidden_dim
        self.seed = seed
        rng = np.random.default_rng(seed)
        limit_1 = np.sqrt(6.0 / (self.input_dim + hidden_dim))
        limit_2 = np.sqrt(6.0 / (hidden_dim + 1))
        self.params: dict[str, np.ndarray] = {
            "w1": rng.uniform(-limit_1, limit_1, (self.input_dim, hidden_dim)),
            "b1": np.zeros(hidden_dim, dtype=np.float64),
            "w2": rng.uniform(-limit_2, limit_2, hidden_dim),
            "b2": np.zeros(1, dtype=np.float64),
        }
        self._adam_m = {
            name: np.zeros_like(value) for name, value in self.params.items()
        }
        self._adam_v = {
            name: np.zeros_like(value) for name, value in self.params.items()
        }
        self._adam_step = 0

    def state_features(self, candidate_features: np.ndarray) -> np.ndarray:
        values = np.asarray(candidate_features, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("value state requires non-empty candidate features")
        if values.shape[1] != self.feature_dim:
            raise ValueError("candidate feature width does not match value network")
        return np.concatenate((np.mean(values, axis=0), np.max(values, axis=0)))

    def forward(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        values = np.asarray(state, dtype=np.float64)
        if values.shape != (self.input_dim,):
            raise ValueError("value state has the wrong shape")
        hidden = np.tanh(values @ self.params["w1"] + self.params["b1"])
        value = float(hidden @ self.params["w2"] + self.params["b2"][0])
        return hidden, value

    def predict(self, state: np.ndarray) -> float:
        return self.forward(state)[1]

    def empty_gradients(self) -> dict[str, np.ndarray]:
        return {name: np.zeros_like(value) for name, value in self.params.items()}

    @staticmethod
    def add_gradients(
        target: dict[str, np.ndarray],
        source: Mapping[str, np.ndarray],
        scale: float = 1.0,
    ) -> None:
        for name in target:
            target[name] += source[name] * scale

    def loss_gradients(
        self,
        state: np.ndarray,
        target: float,
    ) -> tuple[float, float, dict[str, np.ndarray]]:
        hidden, value = self.forward(state)
        error = value - float(target)
        hidden_gradient = error * self.params["w2"] * (1.0 - hidden**2)
        gradients = {
            "w2": hidden * error,
            "b2": np.asarray([error], dtype=np.float64),
            "w1": np.outer(np.asarray(state, dtype=np.float64), hidden_gradient),
            "b1": hidden_gradient,
        }
        return 0.5 * error * error, value, gradients

    def apply_gradients(
        self,
        gradients: Mapping[str, np.ndarray],
        learning_rate: float,
        clip_norm: float,
    ) -> float:
        norm = float(
            np.sqrt(sum(float(np.sum(value * value)) for value in gradients.values()))
        )
        scaled = {name: np.asarray(value) for name, value in gradients.items()}
        if norm > clip_norm > 0:
            factor = clip_norm / (norm + 1e-12)
            scaled = {name: value * factor for name, value in scaled.items()}
        self._adam_step += 1
        beta1, beta2 = 0.9, 0.999
        for name, gradient in scaled.items():
            self._adam_m[name] = beta1 * self._adam_m[name] + (1 - beta1) * gradient
            self._adam_v[name] = beta2 * self._adam_v[name] + (1 - beta2) * (
                gradient * gradient
            )
            m_hat = self._adam_m[name] / (1 - beta1**self._adam_step)
            v_hat = self._adam_v[name] / (1 - beta2**self._adam_step)
            self.params[name] -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
        return norm

    def snapshot(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in self.params.items()}

    def clone(self) -> ValueNetwork:
        clone = ValueNetwork(self.feature_dim, self.hidden_dim, self.seed)
        for name in clone.params:
            clone.params[name] = self.params[name].copy()
        return clone

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            format_version=np.asarray([1], dtype=np.int64),
            feature_dim=np.asarray([self.feature_dim], dtype=np.int64),
            hidden_dim=np.asarray([self.hidden_dim], dtype=np.int64),
            seed=np.asarray([self.seed], dtype=np.int64),
            **self.params,
        )

    @classmethod
    def load(cls, path: str | Path) -> ValueNetwork:
        data = np.load(path, allow_pickle=False)
        if int(data["format_version"][0]) != 1:
            raise ValueError("unsupported value checkpoint format")
        network = cls(
            feature_dim=int(data["feature_dim"][0]),
            hidden_dim=int(data["hidden_dim"][0]),
            seed=int(data["seed"][0]),
        )
        for name in network.params:
            network.params[name] = np.asarray(data[name], dtype=np.float64)
        return network


def value_parameter_hash(network: ValueNetwork) -> str:
    digest = hashlib.sha256()
    schema = {
        "format_version": 1,
        "feature_dim": network.feature_dim,
        "hidden_dim": network.hidden_dim,
        "parameters": [
            {
                "name": name,
                "shape": list(network.params[name].shape),
                "dtype": "float64-little-endian",
            }
            for name in sorted(network.params)
        ],
    }
    digest.update(
        json.dumps(
            schema,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for name in sorted(network.params):
        digest.update(name.encode("utf-8"))
        values = np.asarray(network.params[name], dtype="<f8", order="C")
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class PPOTransition:
    candidate_features: np.ndarray
    selected_index: int
    old_log_probability: float
    state_features: np.ndarray
    advantage: float
    return_value: float


@dataclass(frozen=True)
class TaskGNNPPOTransition:
    frozen_state: FrozenTaskGNNState
    selected_index: int
    old_log_probability: float
    state_features: np.ndarray
    advantage: float
    return_value: float


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(rewards) != len(values) or not rewards:
        raise ValueError("GAE requires equally sized non-empty rewards and values")
    advantages = np.zeros(len(rewards), dtype=np.float64)
    next_value = 0.0
    next_advantage = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        delta = float(rewards[index]) + gamma * next_value - float(values[index])
        next_advantage = delta + gamma * gae_lambda * next_advantage
        advantages[index] = next_advantage
        next_value = float(values[index])
    returns = advantages + np.asarray(values, dtype=np.float64)
    return advantages, returns


def _clone_policy(policy: MaskedMLPPolicy) -> MaskedMLPPolicy:
    clone = MaskedMLPPolicy(
        hidden_dim=policy.hidden_dim,
        seed=policy.seed,
        deterministic=True,
        feature_names=policy.feature_names,
    )
    for name in clone.params:
        clone.params[name] = policy.params[name].copy()
    return clone


@dataclass
class PPOResumeState:
    actor: MaskedMLPPolicy
    critic: ValueNetwork
    rng: np.random.Generator
    best_actor: MaskedMLPPolicy
    best_critic: ValueNetwork
    history: list[dict[str, Any]]
    best_key: tuple[float, float, float]
    best_epoch: int
    completed_epoch: int


@dataclass
class TaskGNNPPOResumeState:
    actor: TaskGNNPolicy
    critic: ValueNetwork
    rng: np.random.Generator
    best_actor: TaskGNNPolicy
    best_critic: ValueNetwork
    history: list[dict[str, Any]]
    best_key: tuple[float, float, float]
    best_epoch: int
    completed_epoch: int


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resume_payload_sha256(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> str:
    digest = hashlib.sha256()
    digest.update(_canonical_sha256(metadata).encode("ascii"))
    for name in sorted(arrays):
        values = np.asarray(arrays[name], dtype="<f8", order="C")
        digest.update(name.encode("utf-8"))
        digest.update(
            json.dumps(list(values.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _resume_array(
    arrays: Mapping[str, np.ndarray],
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    if name not in arrays:
        _fail(
            "ppo_resume_state_corrupt",
            f"$.resume.{name}",
            "training state array is missing",
        )
    try:
        values = np.asarray(arrays[name], dtype=np.float64)
    except (TypeError, ValueError) as error:
        _fail(
            "ppo_resume_state_corrupt",
            f"$.resume.{name}",
            str(error),
        )
    if values.shape != shape or not np.all(np.isfinite(values)):
        _fail(
            "ppo_resume_state_corrupt",
            f"$.resume.{name}",
            "training state array has an invalid shape or non-finite values",
            details={"expected_shape": list(shape), "actual_shape": list(values.shape)},
        )
    return values.copy()


def _save_ppo_resume_state(
    path: Path,
    *,
    contract_sha256: str,
    completed_epoch: int,
    actor: MaskedMLPPolicy,
    critic: ValueNetwork,
    rng: np.random.Generator,
    best_actor: MaskedMLPPolicy,
    best_critic: ValueNetwork,
    history: Sequence[Mapping[str, Any]],
    best_key: tuple[float, float, float],
    best_epoch: int,
) -> None:
    metadata = {
        "format_version": 1,
        "algorithm": "masked_ppo_epoch_boundary_resume",
        "contract_sha256": contract_sha256,
        "completed_epoch": completed_epoch,
        "seed": actor.seed,
        "feature_names": list(actor.feature_names),
        "actor_hidden_dim": actor.hidden_dim,
        "actor_adam_step": actor._adam_step,
        "critic_feature_dim": critic.feature_dim,
        "critic_hidden_dim": critic.hidden_dim,
        "critic_seed": critic.seed,
        "critic_adam_step": critic._adam_step,
        "actor_rng_state": actor.rng.bit_generator.state,
        "training_rng_state": rng.bit_generator.state,
        "history": list(history),
        "best_key": list(best_key),
        "best_epoch": best_epoch,
        "test_accessed": False,
    }
    arrays: dict[str, np.ndarray] = {}
    for name, values in actor.params.items():
        arrays[f"actor_param_{name}"] = values
        arrays[f"actor_adam_m_{name}"] = actor._adam_m[name]
        arrays[f"actor_adam_v_{name}"] = actor._adam_v[name]
        arrays[f"best_actor_param_{name}"] = best_actor.params[name]
    for name, values in critic.params.items():
        arrays[f"critic_param_{name}"] = values
        arrays[f"critic_adam_m_{name}"] = critic._adam_m[name]
        arrays[f"critic_adam_v_{name}"] = critic._adam_v[name]
        arrays[f"best_critic_param_{name}"] = best_critic.params[name]
    metadata["payload_sha256"] = _resume_payload_sha256(metadata, arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                metadata_json=np.asarray(
                    json.dumps(
                        metadata,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                ),
                **arrays,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError) as error:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        _fail(
            "ppo_resume_state_write",
            "$.resume",
            str(error),
        )


def _load_ppo_resume_state(
    path: Path,
    *,
    contract_sha256: str,
    warm_start: MaskedMLPPolicy,
    config: Mapping[str, Any],
) -> PPOResumeState:
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata_text = str(data["metadata_json"].item())
            arrays = {name: np.asarray(data[name]) for name in data.files}
    except (OSError, ValueError, KeyError, EOFError) as error:
        _fail(
            "ppo_resume_state_read",
            "$.resume",
            str(error),
        )
    try:
        metadata = json.loads(metadata_text)
    except json.JSONDecodeError as error:
        _fail(
            "ppo_resume_state_read",
            "$.resume.metadata_json",
            str(error),
        )
    if not isinstance(metadata, dict) or metadata.get("format_version") != 1:
        _fail(
            "ppo_resume_state_version",
            "$.resume.format_version",
            "expected resume state format version 1",
        )
    stored_payload_sha256 = metadata.get("payload_sha256")
    metadata_without_hash = dict(metadata)
    metadata_without_hash.pop("payload_sha256", None)
    numeric_arrays = {
        name: values for name, values in arrays.items() if name != "metadata_json"
    }
    if (
        not isinstance(stored_payload_sha256, str)
        or len(stored_payload_sha256) != 64
        or stored_payload_sha256
        != _resume_payload_sha256(metadata_without_hash, numeric_arrays)
    ):
        _fail(
            "ppo_resume_state_hash",
            "$.resume.payload_sha256",
            "training state payload hash does not match its contents",
        )
    expected_identity = {
        "contract_sha256": contract_sha256,
        "seed": warm_start.seed,
        "feature_names": list(warm_start.feature_names),
        "actor_hidden_dim": warm_start.hidden_dim,
        "critic_feature_dim": len(warm_start.feature_names),
        "critic_hidden_dim": int(config["value_hidden_dim"]),
        "critic_seed": warm_start.seed + 404,
        "test_accessed": False,
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected_identity.items()
        if metadata.get(key) != value
    }
    if mismatches:
        _fail(
            "ppo_resume_state_mismatch",
            "$.resume",
            "resume state does not match the current code, data, config, or warm start",
            details={"mismatches": mismatches},
        )
    completed_epoch = metadata.get("completed_epoch")
    best_epoch = metadata.get("best_epoch")
    actor_step = metadata.get("actor_adam_step")
    critic_step = metadata.get("critic_adam_step")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (completed_epoch, best_epoch, actor_step, critic_step)
    ):
        _fail(
            "ppo_resume_state_corrupt",
            "$.resume",
            "epoch and Adam counters must be non-negative integers",
        )
    assert isinstance(completed_epoch, int)
    assert isinstance(best_epoch, int)
    assert isinstance(actor_step, int)
    assert isinstance(critic_step, int)
    if completed_epoch > int(config["epochs"]) or best_epoch > completed_epoch:
        _fail(
            "ppo_resume_state_mismatch",
            "$.resume.completed_epoch",
            "resume epoch is outside the configured training range",
        )
    history = metadata.get("history")
    if (
        not isinstance(history, list)
        or len(history) != completed_epoch + 1
        or any(
            not isinstance(record, dict) or record.get("epoch") != index
            for index, record in enumerate(history)
        )
    ):
        _fail(
            "ppo_resume_state_corrupt",
            "$.resume.history",
            "history does not match the completed epoch boundary",
        )
    raw_best_key = metadata.get("best_key")
    if (
        not isinstance(raw_best_key, list)
        or len(raw_best_key) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in raw_best_key
        )
    ):
        _fail(
            "ppo_resume_state_corrupt",
            "$.resume.best_key",
            "best selection key is invalid",
        )
    try:
        expected_adam_step = sum(
            len(record["update"]["updates"])
            * math.ceil(
                int(record["update"]["transition_count"])
                / int(config["minibatch_size"])
            )
            for record in history[1:]
        )
    except (KeyError, TypeError, ValueError) as error:
        _fail(
            "ppo_resume_state_corrupt",
            "$.resume.history",
            str(error),
        )
    if actor_step != expected_adam_step or critic_step != expected_adam_step:
        _fail(
            "ppo_resume_state_corrupt",
            "$.resume.actor_adam_step",
            "Adam counters do not match the completed update history",
            details={
                "expected": expected_adam_step,
                "actor": actor_step,
                "critic": critic_step,
            },
        )

    actor = _clone_policy(warm_start)
    critic = ValueNetwork(
        feature_dim=len(actor.feature_names),
        hidden_dim=int(config["value_hidden_dim"]),
        seed=actor.seed + 404,
    )
    best_actor = _clone_policy(actor)
    best_critic = critic.clone()
    for name, values in actor.params.items():
        shape = values.shape
        actor.params[name] = _resume_array(arrays, f"actor_param_{name}", shape)
        actor._adam_m[name] = _resume_array(
            arrays, f"actor_adam_m_{name}", shape
        )
        actor._adam_v[name] = _resume_array(
            arrays, f"actor_adam_v_{name}", shape
        )
        best_actor.params[name] = _resume_array(
            arrays, f"best_actor_param_{name}", shape
        )
    for name, values in critic.params.items():
        shape = values.shape
        critic.params[name] = _resume_array(arrays, f"critic_param_{name}", shape)
        critic._adam_m[name] = _resume_array(
            arrays, f"critic_adam_m_{name}", shape
        )
        critic._adam_v[name] = _resume_array(
            arrays, f"critic_adam_v_{name}", shape
        )
        best_critic.params[name] = _resume_array(
            arrays, f"best_critic_param_{name}", shape
        )
    actor._adam_step = actor_step
    critic._adam_step = critic_step
    rng = np.random.default_rng()
    try:
        actor.rng.bit_generator.state = metadata["actor_rng_state"]
        rng.bit_generator.state = metadata["training_rng_state"]
    except (KeyError, TypeError, ValueError) as error:
        _fail(
            "ppo_resume_state_corrupt",
            "$.resume.rng_state",
            str(error),
        )
    return PPOResumeState(
        actor=actor,
        critic=critic,
        rng=rng,
        best_actor=best_actor,
        best_critic=best_critic,
        history=history,
        best_key=tuple(float(value) for value in raw_best_key),
        best_epoch=best_epoch,
        completed_epoch=completed_epoch,
    )


def _save_task_gnn_ppo_resume_state(
    path: Path,
    *,
    contract_sha256: str,
    warm_start_sha256: str,
    completed_epoch: int,
    actor: TaskGNNPolicy,
    critic: ValueNetwork,
    rng: np.random.Generator,
    best_actor: TaskGNNPolicy,
    best_critic: ValueNetwork,
    history: Sequence[Mapping[str, Any]],
    best_key: tuple[float, float, float],
    best_epoch: int,
) -> None:
    metadata = {
        "format_version": 1,
        "algorithm": "task_gnn_masked_ppo_epoch_boundary_resume",
        "architecture": actor.architecture,
        "contract_sha256": contract_sha256,
        "warm_start_parameter_sha256": warm_start_sha256,
        "completed_epoch": completed_epoch,
        "seed": actor.seed,
        "feature_names": list(actor.feature_names),
        "actor_hidden_dim": actor.hidden_dim,
        "actor_message_dim": actor.message_dim,
        "actor_parameter_names": sorted(actor.params),
        "actor_adam_step": actor._adam_step,
        "critic_feature_dim": critic.feature_dim,
        "critic_hidden_dim": critic.hidden_dim,
        "critic_seed": critic.seed,
        "critic_adam_step": critic._adam_step,
        "actor_rng_state": actor.rng.bit_generator.state,
        "training_rng_state": rng.bit_generator.state,
        "history": list(history),
        "best_key": list(best_key),
        "best_epoch": best_epoch,
        "test_accessed": False,
    }
    arrays: dict[str, np.ndarray] = {}
    for name, values in actor.params.items():
        arrays[f"actor_param_{name}"] = values
        arrays[f"actor_adam_m_{name}"] = actor._adam_m[name]
        arrays[f"actor_adam_v_{name}"] = actor._adam_v[name]
        arrays[f"best_actor_param_{name}"] = best_actor.params[name]
    for name, values in critic.params.items():
        arrays[f"critic_param_{name}"] = values
        arrays[f"critic_adam_m_{name}"] = critic._adam_m[name]
        arrays[f"critic_adam_v_{name}"] = critic._adam_v[name]
        arrays[f"best_critic_param_{name}"] = best_critic.params[name]
    metadata["payload_sha256"] = _resume_payload_sha256(metadata, arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                metadata_json=np.asarray(
                    json.dumps(
                        metadata,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                ),
                **arrays,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError) as error:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        _fail("ppo_resume_state_write", "$.resume", str(error))


def _load_task_gnn_ppo_resume_state(
    path: Path,
    *,
    contract_sha256: str,
    warm_start: TaskGNNPolicy,
    config: Mapping[str, Any],
) -> TaskGNNPPOResumeState:
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata_text = str(data["metadata_json"].item())
            arrays = {name: np.asarray(data[name]) for name in data.files}
    except (OSError, ValueError, KeyError, EOFError) as error:
        _fail("ppo_resume_state_read", "$.resume", str(error))
    try:
        metadata = json.loads(metadata_text)
    except json.JSONDecodeError as error:
        _fail("ppo_resume_state_read", "$.resume.metadata_json", str(error))
    if not isinstance(metadata, dict) or metadata.get("format_version") != 1:
        _fail(
            "ppo_resume_state_version",
            "$.resume.format_version",
            "expected task-GNN resume state format version 1",
        )
    stored_payload_sha256 = metadata.get("payload_sha256")
    metadata_without_hash = dict(metadata)
    metadata_without_hash.pop("payload_sha256", None)
    numeric_arrays = {
        name: values for name, values in arrays.items() if name != "metadata_json"
    }
    if (
        not isinstance(stored_payload_sha256, str)
        or len(stored_payload_sha256) != 64
        or stored_payload_sha256
        != _resume_payload_sha256(metadata_without_hash, numeric_arrays)
    ):
        _fail(
            "ppo_resume_state_hash",
            "$.resume.payload_sha256",
            "task-GNN training state payload hash does not match its contents",
        )
    expected_identity = {
        "algorithm": "task_gnn_masked_ppo_epoch_boundary_resume",
        "architecture": warm_start.architecture,
        "contract_sha256": contract_sha256,
        "warm_start_parameter_sha256": task_gnn_parameter_hash(warm_start),
        "seed": warm_start.seed,
        "feature_names": list(warm_start.feature_names),
        "actor_hidden_dim": warm_start.hidden_dim,
        "actor_message_dim": warm_start.message_dim,
        "actor_parameter_names": sorted(warm_start.params),
        "critic_feature_dim": len(warm_start.feature_names),
        "critic_hidden_dim": int(config["value_hidden_dim"]),
        "critic_seed": warm_start.seed + 404,
        "test_accessed": False,
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected_identity.items()
        if metadata.get(key) != value
    }
    if mismatches:
        _fail(
            "ppo_resume_state_mismatch",
            "$.resume",
            "task-GNN resume state does not match the current contract",
            details={"mismatches": mismatches},
        )
    completed_epoch = metadata.get("completed_epoch")
    best_epoch = metadata.get("best_epoch")
    actor_step = metadata.get("actor_adam_step")
    critic_step = metadata.get("critic_adam_step")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (completed_epoch, best_epoch, actor_step, critic_step)
    ):
        _fail(
            "ppo_resume_state_corrupt",
            "$.resume",
            "epoch and Adam counters must be non-negative integers",
        )
    assert isinstance(completed_epoch, int)
    assert isinstance(best_epoch, int)
    assert isinstance(actor_step, int)
    assert isinstance(critic_step, int)
    if completed_epoch > int(config["epochs"]) or best_epoch > completed_epoch:
        _fail(
            "ppo_resume_state_mismatch",
            "$.resume.completed_epoch",
            "task-GNN resume epoch is outside the training range",
        )
    history = metadata.get("history")
    if (
        not isinstance(history, list)
        or len(history) != completed_epoch + 1
        or any(
            not isinstance(record, dict) or record.get("epoch") != index
            for index, record in enumerate(history)
        )
    ):
        _fail(
            "ppo_resume_state_corrupt",
            "$.resume.history",
            "task-GNN history does not match the completed epoch boundary",
        )
    raw_best_key = metadata.get("best_key")
    if (
        not isinstance(raw_best_key, list)
        or len(raw_best_key) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in raw_best_key
        )
    ):
        _fail(
            "ppo_resume_state_corrupt",
            "$.resume.best_key",
            "task-GNN best selection key is invalid",
        )
    try:
        expected_adam_step = sum(
            len(record["update"]["updates"])
            * math.ceil(
                int(record["update"]["transition_count"])
                / int(config["minibatch_size"])
            )
            for record in history[1:]
        )
    except (KeyError, TypeError, ValueError) as error:
        _fail("ppo_resume_state_corrupt", "$.resume.history", str(error))
    if actor_step != expected_adam_step or critic_step != expected_adam_step:
        _fail(
            "ppo_resume_state_corrupt",
            "$.resume.actor_adam_step",
            "task-GNN Adam counters do not match update history",
            details={
                "expected": expected_adam_step,
                "actor": actor_step,
                "critic": critic_step,
            },
        )
    actor = warm_start.clone(include_optimizer=False)
    critic = ValueNetwork(
        feature_dim=len(actor.feature_names),
        hidden_dim=int(config["value_hidden_dim"]),
        seed=actor.seed + 404,
    )
    expected_array_names = {
        key
        for name in actor.params
        for key in (
            f"actor_param_{name}",
            f"actor_adam_m_{name}",
            f"actor_adam_v_{name}",
            f"best_actor_param_{name}",
        )
    } | {
        key
        for name in critic.params
        for key in (
            f"critic_param_{name}",
            f"critic_adam_m_{name}",
            f"critic_adam_v_{name}",
            f"best_critic_param_{name}",
        )
    }
    if set(numeric_arrays) != expected_array_names:
        _fail(
            "ppo_resume_state_corrupt",
            "$.resume.arrays",
            "task-GNN state array schema does not match the architecture",
            details={
                "missing": sorted(expected_array_names - set(numeric_arrays)),
                "unexpected": sorted(set(numeric_arrays) - expected_array_names),
            },
        )
    best_actor = actor.clone(include_optimizer=False)
    best_critic = critic.clone()
    for name, values in actor.params.items():
        shape = values.shape
        actor.params[name] = _resume_array(arrays, f"actor_param_{name}", shape)
        actor._adam_m[name] = _resume_array(
            arrays,
            f"actor_adam_m_{name}",
            shape,
        )
        actor._adam_v[name] = _resume_array(
            arrays,
            f"actor_adam_v_{name}",
            shape,
        )
        best_actor.params[name] = _resume_array(
            arrays,
            f"best_actor_param_{name}",
            shape,
        )
    for name, values in critic.params.items():
        shape = values.shape
        critic.params[name] = _resume_array(arrays, f"critic_param_{name}", shape)
        critic._adam_m[name] = _resume_array(
            arrays,
            f"critic_adam_m_{name}",
            shape,
        )
        critic._adam_v[name] = _resume_array(
            arrays,
            f"critic_adam_v_{name}",
            shape,
        )
        best_critic.params[name] = _resume_array(
            arrays,
            f"best_critic_param_{name}",
            shape,
        )
    actor._adam_step = actor_step
    critic._adam_step = critic_step
    rng = np.random.default_rng()
    try:
        actor.rng.bit_generator.state = metadata["actor_rng_state"]
        rng.bit_generator.state = metadata["training_rng_state"]
    except (KeyError, TypeError, ValueError) as error:
        _fail("ppo_resume_state_corrupt", "$.resume.rng_state", str(error))
    return TaskGNNPPOResumeState(
        actor=actor,
        critic=critic,
        rng=rng,
        best_actor=best_actor,
        best_critic=best_critic,
        history=history,
        best_key=tuple(float(value) for value in raw_best_key),
        best_epoch=best_epoch,
        completed_epoch=completed_epoch,
    )


def _manifest_records(
    manifest: Mapping[str, Any],
    scenarios: Sequence[Scenario],
    *,
    split: str,
    purpose: str,
) -> dict[str, Mapping[str, Any]]:
    if manifest.get("split") != split or manifest.get("purpose") != purpose:
        _fail("ppo_split_usage", "$.split", "manifest split or purpose changed")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        _fail("teacher_manifest", "$.entries", "expected an array")
    records = {
        str(entry["scenario_id"]): entry
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("scenario_id"), str)
    }
    if len(records) != len(entries) or set(records) != {
        scenario.id for scenario in scenarios
    }:
        _fail(
            "teacher_manifest",
            "$.entries",
            "manifest scenarios do not exactly match PPO scenarios",
        )
    for scenario in scenarios:
        if records[scenario.id].get("scenario_hash") != scenario.content_hash():
            _fail(
                "teacher_manifest",
                f"$.entries.{scenario.id}",
                "Scenario hash changed",
            )
    return records


def _collect_episode(
    actor: MaskedMLPPolicy,
    critic: ValueNetwork,
    scenario: Scenario,
    heft_makespan: float,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[list[PPOTransition], float, float]:
    env = HeterogeneousDagEnv(scenario)
    actor.reset(scenario)
    features: list[np.ndarray] = []
    selected_indices: list[int] = []
    old_log_probabilities: list[float] = []
    states: list[np.ndarray] = []
    values: list[float] = []
    rewards: list[float] = []
    current_makespan = 0.0
    while not env.done:
        cache = actor.distribution(env)
        selected_index = int(
            actor.rng.choice(len(cache.actions), p=cache.probabilities)
        )
        action = cache.actions[selected_index]
        state = critic.state_features(cache.features)
        value = critic.predict(state)
        env.step(*action)
        next_makespan = max(entry.finish for entry in env.entries.values())
        reward = -(next_makespan - current_makespan) / heft_makespan
        current_makespan = next_makespan
        features.append(cache.features.copy())
        selected_indices.append(selected_index)
        old_log_probabilities.append(
            float(np.log(cache.probabilities[selected_index] + 1e-12))
        )
        states.append(state)
        values.append(value)
        rewards.append(reward)
    result = env.result(actor.name)
    validate_schedule(scenario, result)
    validate_schedule_independent(scenario, result)
    ratio = result.makespan / heft_makespan
    reward_error = abs(float(np.sum(rewards)) + ratio)
    if reward_error > 1e-9:
        raise RuntimeError("incremental PPO rewards do not sum to negative ratio")
    advantages, returns = compute_gae(
        rewards,
        values,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    transitions = [
        PPOTransition(
            candidate_features=features[index],
            selected_index=selected_indices[index],
            old_log_probability=old_log_probabilities[index],
            state_features=states[index],
            advantage=float(advantages[index]),
            return_value=float(returns[index]),
        )
        for index in range(len(rewards))
    ]
    return transitions, ratio, reward_error


def _update_ppo(
    actor: MaskedMLPPolicy,
    critic: ValueNetwork,
    transitions: Sequence[PPOTransition],
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    if not transitions:
        raise ValueError("PPO update requires transitions")
    raw_advantages = np.asarray(
        [transition.advantage for transition in transitions],
        dtype=np.float64,
    )
    advantage_mean = float(np.mean(raw_advantages))
    advantage_std = float(np.std(raw_advantages))
    normalized_advantages = (raw_advantages - advantage_mean) / (
        advantage_std + 1e-8
    )
    clip_ratio = float(config["clip_ratio"])
    entropy_coefficient = float(config["entropy_coefficient"])
    update_records: list[dict[str, Any]] = []
    stopped_early = False
    for update_epoch in range(1, int(config["update_epochs"]) + 1):
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        approximate_kls: list[float] = []
        clipped: list[float] = []
        actor_norms: list[float] = []
        value_norms: list[float] = []
        permutation = rng.permutation(len(transitions))
        for start in range(0, len(transitions), int(config["minibatch_size"])):
            indices = permutation[
                start : start + int(config["minibatch_size"])
            ]
            if len(indices) == 0:
                continue
            actor_gradients = actor.empty_gradients()
            value_gradients = critic.empty_gradients()
            for raw_index in indices:
                index = int(raw_index)
                transition = transitions[index]
                advantage = float(normalized_advantages[index])
                cache = actor.distribution_from_features(
                    transition.candidate_features
                )
                probability = float(cache.probabilities[transition.selected_index])
                new_log_probability = float(np.log(probability + 1e-12))
                log_ratio = new_log_probability - transition.old_log_probability
                ratio = float(np.exp(np.clip(log_ratio, -20.0, 20.0)))
                unclipped_objective = ratio * advantage
                clipped_objective = (
                    float(np.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio))
                    * advantage
                )
                objective = min(unclipped_objective, clipped_objective)
                policy_losses.append(-objective)
                is_clipped = unclipped_objective > clipped_objective + 1e-15
                clipped.append(float(is_clipped))
                if not is_clipped:
                    policy_gradient = actor.log_probability_gradients(
                        cache,
                        transition.selected_index,
                    )
                    actor.add_gradients(
                        actor_gradients,
                        policy_gradient,
                        scale=ratio * advantage,
                    )
                entropy = -float(
                    np.sum(
                        cache.probabilities
                        * np.log(cache.probabilities + 1e-12)
                    )
                )
                entropies.append(entropy)
                if entropy_coefficient:
                    actor.add_gradients(
                        actor_gradients,
                        actor.entropy_gradients(cache),
                        scale=entropy_coefficient,
                    )
                value_loss, _, gradients = critic.loss_gradients(
                    transition.state_features,
                    transition.return_value,
                )
                value_losses.append(value_loss)
                critic.add_gradients(value_gradients, gradients)
                approximate_kls.append(0.5 * log_ratio * log_ratio)
            scale = 1.0 / len(indices)
            actor_gradients = {
                name: gradient * scale
                for name, gradient in actor_gradients.items()
            }
            value_gradients = {
                name: gradient * scale
                for name, gradient in value_gradients.items()
            }
            actor_norms.append(
                actor.apply_gradients(
                    actor_gradients,
                    float(config["actor_learning_rate"]),
                    float(config["gradient_clip"]),
                )
            )
            value_norms.append(
                critic.apply_gradients(
                    value_gradients,
                    float(config["value_learning_rate"]),
                    float(config["gradient_clip"]),
                )
            )
        record = {
            "update_epoch": update_epoch,
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "approximate_kl": float(np.mean(approximate_kls)),
            "clip_fraction": float(np.mean(clipped)),
            "mean_actor_gradient_norm": float(np.mean(actor_norms)),
            "mean_value_gradient_norm": float(np.mean(value_norms)),
        }
        update_records.append(record)
        if record["approximate_kl"] > float(config["target_kl"]):
            stopped_early = True
            break
    return {
        "transition_count": len(transitions),
        "advantage_mean_before_normalization": advantage_mean,
        "advantage_std_before_normalization": advantage_std,
        "stopped_early_for_kl": stopped_early,
        "updates": update_records,
    }


def _collect_task_gnn_episode(
    actor: TaskGNNPolicy,
    critic: ValueNetwork,
    scenario: Scenario,
    heft_makespan: float,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[list[TaskGNNPPOTransition], float, float]:
    env = HeterogeneousDagEnv(scenario)
    graph = freeze_task_graph(scenario)
    frozen_states: list[FrozenTaskGNNState] = []
    selected_indices: list[int] = []
    old_log_probabilities: list[float] = []
    value_states: list[np.ndarray] = []
    values: list[float] = []
    rewards: list[float] = []
    current_makespan = 0.0
    while not env.done:
        frozen_state = freeze_task_gnn_state(env, graph=graph)
        cache = actor.distribution_from_frozen_state(frozen_state)
        selected_index = int(
            actor.rng.choice(len(cache.actions), p=cache.probabilities)
        )
        action = cache.actions[selected_index]
        value_state = critic.state_features(cache.features)
        value_state.setflags(write=False)
        value = critic.predict(value_state)
        env.step(*action)
        next_makespan = max(entry.finish for entry in env.entries.values())
        reward = -(next_makespan - current_makespan) / heft_makespan
        current_makespan = next_makespan
        frozen_states.append(frozen_state)
        selected_indices.append(selected_index)
        old_log_probabilities.append(
            float(np.log(cache.probabilities[selected_index] + 1e-12))
        )
        value_states.append(value_state)
        values.append(value)
        rewards.append(reward)
    result = env.result(actor.name)
    validate_schedule(scenario, result)
    validate_schedule_independent(scenario, result)
    ratio = result.makespan / heft_makespan
    reward_error = abs(float(np.sum(rewards)) + ratio)
    if reward_error > 1e-9:
        raise RuntimeError(
            "incremental task-GNN PPO rewards do not sum to negative ratio"
        )
    advantages, returns = compute_gae(
        rewards,
        values,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    transitions = [
        TaskGNNPPOTransition(
            frozen_state=frozen_states[index],
            selected_index=selected_indices[index],
            old_log_probability=old_log_probabilities[index],
            state_features=value_states[index],
            advantage=float(advantages[index]),
            return_value=float(returns[index]),
        )
        for index in range(len(rewards))
    ]
    return transitions, ratio, reward_error


def _update_task_gnn_ppo(
    actor: TaskGNNPolicy,
    critic: ValueNetwork,
    transitions: Sequence[TaskGNNPPOTransition],
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    if not transitions:
        raise ValueError("task-GNN PPO update requires transitions")
    raw_advantages = np.asarray(
        [transition.advantage for transition in transitions],
        dtype=np.float64,
    )
    advantage_mean = float(np.mean(raw_advantages))
    advantage_std = float(np.std(raw_advantages))
    normalized_advantages = (raw_advantages - advantage_mean) / (
        advantage_std + 1e-8
    )
    clip_ratio = float(config["clip_ratio"])
    entropy_coefficient = float(config["entropy_coefficient"])
    update_records: list[dict[str, Any]] = []
    stopped_early = False
    for update_epoch in range(1, int(config["update_epochs"]) + 1):
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        approximate_kls: list[float] = []
        clipped: list[float] = []
        actor_norms: list[float] = []
        value_norms: list[float] = []
        permutation = rng.permutation(len(transitions))
        for start in range(0, len(transitions), int(config["minibatch_size"])):
            indices = permutation[
                start : start + int(config["minibatch_size"])
            ]
            if len(indices) == 0:
                continue
            actor_gradients = actor.empty_gradients()
            value_gradients = critic.empty_gradients()
            for raw_index in indices:
                index = int(raw_index)
                transition = transitions[index]
                advantage = float(normalized_advantages[index])
                cache = actor.distribution_from_frozen_state(
                    transition.frozen_state
                )
                probability = float(cache.probabilities[transition.selected_index])
                new_log_probability = float(np.log(probability + 1e-12))
                log_ratio = new_log_probability - transition.old_log_probability
                ratio = float(np.exp(np.clip(log_ratio, -20.0, 20.0)))
                unclipped_objective = ratio * advantage
                clipped_objective = (
                    float(np.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio))
                    * advantage
                )
                objective = min(unclipped_objective, clipped_objective)
                policy_losses.append(-objective)
                is_clipped = unclipped_objective > clipped_objective + 1e-15
                clipped.append(float(is_clipped))
                if not is_clipped:
                    policy_gradient = actor.log_probability_gradients(
                        cache,
                        transition.selected_index,
                    )
                    actor.add_gradients(
                        actor_gradients,
                        policy_gradient,
                        scale=ratio * advantage,
                    )
                entropy = -float(
                    np.sum(
                        cache.probabilities
                        * np.log(cache.probabilities + 1e-12)
                    )
                )
                entropies.append(entropy)
                if entropy_coefficient:
                    actor.add_gradients(
                        actor_gradients,
                        actor.entropy_gradients(cache),
                        scale=entropy_coefficient,
                    )
                value_loss, _, gradients = critic.loss_gradients(
                    transition.state_features,
                    transition.return_value,
                )
                value_losses.append(value_loss)
                critic.add_gradients(value_gradients, gradients)
                approximate_kls.append(0.5 * log_ratio * log_ratio)
            scale = 1.0 / len(indices)
            actor_gradients = {
                name: gradient * scale
                for name, gradient in actor_gradients.items()
            }
            value_gradients = {
                name: gradient * scale
                for name, gradient in value_gradients.items()
            }
            actor_norms.append(
                actor.apply_gradients(
                    actor_gradients,
                    float(config["actor_learning_rate"]),
                    float(config["gradient_clip"]),
                )
            )
            value_norms.append(
                critic.apply_gradients(
                    value_gradients,
                    float(config["value_learning_rate"]),
                    float(config["gradient_clip"]),
                )
            )
        record = {
            "update_epoch": update_epoch,
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "approximate_kl": float(np.mean(approximate_kls)),
            "clip_fraction": float(np.mean(clipped)),
            "mean_actor_gradient_norm": float(np.mean(actor_norms)),
            "mean_value_gradient_norm": float(np.mean(value_norms)),
        }
        update_records.append(record)
        if record["approximate_kl"] > float(config["target_kl"]):
            stopped_early = True
            break
    return {
        "transition_count": len(transitions),
        "advantage_mean_before_normalization": advantage_mean,
        "advantage_std_before_normalization": advantage_std,
        "stopped_early_for_kl": stopped_early,
        "updates": update_records,
    }


def train_task_gnn_ppo(
    warm_start: TaskGNNPolicy,
    train_scenarios: Sequence[Scenario],
    train_teacher_manifest: Mapping[str, Any],
    validation_scenarios: Sequence[Scenario],
    validation_reference_manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    seed: int,
    validation_frozen_states: (
        Mapping[str, FrozenTaskGNNTeacherStates] | None
    ) = None,
    resume_state_path: str | Path | None = None,
    resume_contract: Mapping[str, Any] | None = None,
    resume: bool = False,
) -> tuple[
    TaskGNNPolicy,
    TaskGNNPolicy,
    ValueNetwork,
    ValueNetwork,
    dict[str, Any],
]:
    if warm_start.seed != seed:
        _fail(
            "ppo_seed_mismatch",
            "$.seed",
            "task-GNN warm-start seed does not match the PPO seed",
        )
    if abs(float(config["gamma"]) - 1.0) > 1e-12:
        _fail(
            "ppo_reward_contract",
            "$.ppo.gamma",
            "gamma must remain 1.0 for task-GNN reward identity",
        )
    if int(config["episodes_per_epoch"]) > len(train_scenarios):
        _fail(
            "config_value",
            "$.ppo.episodes_per_epoch",
            "cannot exceed the number of train scenarios",
        )
    train_records = _manifest_records(
        train_teacher_manifest,
        train_scenarios,
        split="train",
        purpose="behavior_cloning_teacher",
    )
    _manifest_records(
        validation_reference_manifest,
        validation_scenarios,
        split="validation",
        purpose="model_selection_reference",
    )
    validation_states = (
        freeze_task_gnn_teacher_dataset(
            validation_scenarios,
            validation_reference_manifest,
            split="validation",
            purpose="model_selection_reference",
        )
        if validation_frozen_states is None
        else dict(validation_frozen_states)
    )
    if set(validation_states) != {
        scenario.id for scenario in validation_scenarios
    }:
        _fail(
            "validation_reference",
            "$.entries",
            "frozen task-GNN validation states do not match scenarios",
        )
    state_path = Path(resume_state_path) if resume_state_path is not None else None
    if state_path is not None and resume_contract is None:
        raise ValueError(
            "resume_contract is required when saving task-GNN PPO state"
        )
    if resume and state_path is None:
        _fail(
            "ppo_resume_state_missing",
            "$.resume",
            "task-GNN resume requires a training state path",
        )
    contract_sha256 = (
        _canonical_sha256(resume_contract) if resume_contract is not None else ""
    )
    warm_start_sha256 = task_gnn_parameter_hash(warm_start)
    if resume:
        assert state_path is not None
        if not state_path.is_file():
            _fail(
                "ppo_resume_state_missing",
                "$.resume",
                f"task-GNN training state does not exist: {state_path}",
            )
        restored = _load_task_gnn_ppo_resume_state(
            state_path,
            contract_sha256=contract_sha256,
            warm_start=warm_start,
            config=config,
        )
        actor = restored.actor
        critic = restored.critic
        rng = restored.rng
        best_actor = restored.best_actor
        best_critic = restored.best_critic
        history = restored.history
        best_key = restored.best_key
        best_epoch = restored.best_epoch
        completed_epoch = restored.completed_epoch
    else:
        if state_path is not None and state_path.exists():
            _fail(
                "ppo_resume_state_exists",
                "$.resume",
                f"refusing to overwrite task-GNN state: {state_path}",
            )
        actor = warm_start.clone(include_optimizer=False)
        critic = ValueNetwork(
            feature_dim=len(actor.feature_names),
            hidden_dim=int(config["value_hidden_dim"]),
            seed=seed + 404,
        )
        rng = np.random.default_rng(seed + int(config["shuffle_seed_offset"]))
        initial_validation, _ = evaluate_bc_policy(
            actor,
            validation_scenarios,
            validation_reference_manifest,
            failure_penalty_ratio=float(config["failure_penalty_ratio"]),
            frozen_task_gnn_states=validation_states,
        )
        best_key = (
            float(initial_validation["failure_count"]),
            float(initial_validation["mean_ratio"]),
            0.0,
        )
        best_epoch = 0
        best_actor = actor.clone(include_optimizer=False)
        best_critic = critic.clone()
        history = [
            {
                "epoch": 0,
                "source": "task_gnn_behavior_cloning_warm_start",
                "validation": initial_validation,
            }
        ]
        completed_epoch = 0
        if state_path is not None:
            _save_task_gnn_ppo_resume_state(
                state_path,
                contract_sha256=contract_sha256,
                warm_start_sha256=warm_start_sha256,
                completed_epoch=completed_epoch,
                actor=actor,
                critic=critic,
                rng=rng,
                best_actor=best_actor,
                best_critic=best_critic,
                history=history,
                best_key=best_key,
                best_epoch=best_epoch,
            )
    for epoch in range(completed_epoch + 1, int(config["epochs"]) + 1):
        transitions: list[TaskGNNPPOTransition] = []
        train_ratios: list[float] = []
        reward_errors: list[float] = []
        selected = rng.permutation(len(train_scenarios))[
            : int(config["episodes_per_epoch"])
        ]
        for raw_index in selected:
            scenario = train_scenarios[int(raw_index)]
            episode, ratio, reward_error = _collect_task_gnn_episode(
                actor,
                critic,
                scenario,
                float(train_records[scenario.id]["makespan"]),
                gamma=float(config["gamma"]),
                gae_lambda=float(config["gae_lambda"]),
            )
            transitions.extend(episode)
            train_ratios.append(ratio)
            reward_errors.append(reward_error)
        update = _update_task_gnn_ppo(actor, critic, transitions, config, rng)
        validation, _ = evaluate_bc_policy(
            actor,
            validation_scenarios,
            validation_reference_manifest,
            failure_penalty_ratio=float(config["failure_penalty_ratio"]),
            frozen_task_gnn_states=validation_states,
        )
        history.append(
            {
                "epoch": epoch,
                "source": "task_gnn_masked_ppo",
                "train_episode_count": len(train_ratios),
                "train_mean_ratio": float(np.mean(train_ratios)),
                "train_min_ratio": float(np.min(train_ratios)),
                "train_max_ratio": float(np.max(train_ratios)),
                "reward_identity_max_abs_error": float(
                    np.max(reward_errors)
                ),
                "update": update,
                "validation": validation,
            }
        )
        key = (
            float(validation["failure_count"]),
            float(validation["mean_ratio"]),
            float(epoch),
        )
        if key < best_key:
            best_key = key
            best_epoch = epoch
            best_actor = actor.clone(include_optimizer=False)
            best_critic = critic.clone()
        if state_path is not None:
            _save_task_gnn_ppo_resume_state(
                state_path,
                contract_sha256=contract_sha256,
                warm_start_sha256=warm_start_sha256,
                completed_epoch=epoch,
                actor=actor,
                critic=critic,
                rng=rng,
                best_actor=best_actor,
                best_critic=best_critic,
                history=history,
                best_key=best_key,
                best_epoch=best_epoch,
            )
    return (
        best_actor,
        actor.clone(include_optimizer=False),
        best_critic,
        critic.clone(),
        {
            "format_version": 1,
            "algorithm": "task-GNN masked PPO with GAE and BC warm start",
            "architecture": actor.architecture,
            "seed": seed,
            "feature_names": list(actor.feature_names),
            "reward": "negative incremental makespan divided by HEFT makespan",
            "reward_sum_identity": "sum(step_reward) == -final_ratio",
            "epochs": history,
            "selection": {
                "split": "validation",
                "test_accessed": False,
                "metric": "validation_mean_ratio",
                "candidates": "task-GNN BC warm start plus every PPO epoch",
                "tie_break": [
                    "zero_failures",
                    "lower_mean_ratio",
                    "earlier_epoch",
                ],
                "best_epoch": best_epoch,
            },
        },
    )


def train_masked_ppo(
    warm_start: MaskedMLPPolicy,
    train_scenarios: Sequence[Scenario],
    train_teacher_manifest: Mapping[str, Any],
    validation_scenarios: Sequence[Scenario],
    validation_reference_manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    seed: int,
    validation_frozen_states: Mapping[str, FrozenTeacherStates] | None = None,
    resume_state_path: str | Path | None = None,
    resume_contract: Mapping[str, Any] | None = None,
    resume: bool = False,
) -> tuple[
    MaskedMLPPolicy,
    MaskedMLPPolicy,
    ValueNetwork,
    ValueNetwork,
    dict[str, Any],
]:
    if warm_start.seed != seed:
        _fail(
            "ppo_resume_state_mismatch",
            "$.seed",
            "warm-start seed does not match the PPO seed",
            details={"warm_start_seed": warm_start.seed, "ppo_seed": seed},
        )
    if set(warm_start.feature_names) & set(TEACHER_FEATURE_NAMES):
        _fail(
            "ppo_teacher_feature_leakage",
            "$.feature_names",
            "PPO actor contains direct HEFT decision features",
        )
    if int(config["episodes_per_epoch"]) > len(train_scenarios):
        _fail(
            "config_value",
            "$.ppo.episodes_per_epoch",
            "cannot exceed the number of train scenarios",
        )
    train_records = _manifest_records(
        train_teacher_manifest,
        train_scenarios,
        split="train",
        purpose="behavior_cloning_teacher",
    )
    _manifest_records(
        validation_reference_manifest,
        validation_scenarios,
        split="validation",
        purpose="model_selection_reference",
    )
    state_path = Path(resume_state_path) if resume_state_path is not None else None
    if state_path is not None and resume_contract is None:
        raise ValueError("resume_contract is required when saving PPO training state")
    if resume and state_path is None:
        _fail(
            "ppo_resume_state_missing",
            "$.resume",
            "resume requires a training state path",
        )
    contract_sha256 = (
        _canonical_sha256(resume_contract) if resume_contract is not None else ""
    )
    if resume:
        assert state_path is not None
        if not state_path.is_file():
            _fail(
                "ppo_resume_state_missing",
                "$.resume",
                f"training state does not exist: {state_path}",
            )
        restored = _load_ppo_resume_state(
            state_path,
            contract_sha256=contract_sha256,
            warm_start=warm_start,
            config=config,
        )
        actor = restored.actor
        critic = restored.critic
        rng = restored.rng
        best_actor = restored.best_actor
        best_critic = restored.best_critic
        history = restored.history
        best_key = restored.best_key
        best_epoch = restored.best_epoch
        completed_epoch = restored.completed_epoch
    else:
        if state_path is not None and state_path.exists():
            _fail(
                "ppo_resume_state_exists",
                "$.resume",
                f"refusing to overwrite existing training state: {state_path}",
            )
        actor = _clone_policy(warm_start)
        critic = ValueNetwork(
            feature_dim=len(actor.feature_names),
            hidden_dim=int(config["value_hidden_dim"]),
            seed=seed + 404,
        )
        rng = np.random.default_rng(seed + int(config["shuffle_seed_offset"]))
        initial_validation, _ = evaluate_bc_policy(
            actor,
            validation_scenarios,
            validation_reference_manifest,
            failure_penalty_ratio=float(config["failure_penalty_ratio"]),
            frozen_teacher_states=validation_frozen_states,
        )
        best_key = (
            float(initial_validation["failure_count"]),
            float(initial_validation["mean_ratio"]),
            0.0,
        )
        best_epoch = 0
        best_actor = _clone_policy(actor)
        best_critic = critic.clone()
        history = [
            {
                "epoch": 0,
                "source": "behavior_cloning_warm_start",
                "validation": initial_validation,
            }
        ]
        completed_epoch = 0
        if state_path is not None:
            _save_ppo_resume_state(
                state_path,
                contract_sha256=contract_sha256,
                completed_epoch=completed_epoch,
                actor=actor,
                critic=critic,
                rng=rng,
                best_actor=best_actor,
                best_critic=best_critic,
                history=history,
                best_key=best_key,
                best_epoch=best_epoch,
            )
    for epoch in range(completed_epoch + 1, int(config["epochs"]) + 1):
        transitions: list[PPOTransition] = []
        train_ratios: list[float] = []
        reward_errors: list[float] = []
        selected = rng.permutation(len(train_scenarios))[
            : int(config["episodes_per_epoch"])
        ]
        for raw_index in selected:
            scenario = train_scenarios[int(raw_index)]
            episode, ratio, reward_error = _collect_episode(
                actor,
                critic,
                scenario,
                float(train_records[scenario.id]["makespan"]),
                gamma=float(config["gamma"]),
                gae_lambda=float(config["gae_lambda"]),
            )
            transitions.extend(episode)
            train_ratios.append(ratio)
            reward_errors.append(reward_error)
        update = _update_ppo(actor, critic, transitions, config, rng)
        validation, _ = evaluate_bc_policy(
            actor,
            validation_scenarios,
            validation_reference_manifest,
            failure_penalty_ratio=float(config["failure_penalty_ratio"]),
            frozen_teacher_states=validation_frozen_states,
        )
        history.append(
            {
                "epoch": epoch,
                "source": "masked_ppo",
                "train_episode_count": len(train_ratios),
                "train_mean_ratio": float(np.mean(train_ratios)),
                "train_min_ratio": float(np.min(train_ratios)),
                "train_max_ratio": float(np.max(train_ratios)),
                "reward_identity_max_abs_error": float(np.max(reward_errors)),
                "update": update,
                "validation": validation,
            }
        )
        key = (
            float(validation["failure_count"]),
            float(validation["mean_ratio"]),
            float(epoch),
        )
        if key < best_key:
            best_key = key
            best_epoch = epoch
            best_actor = _clone_policy(actor)
            best_critic = critic.clone()
        if state_path is not None:
            _save_ppo_resume_state(
                state_path,
                contract_sha256=contract_sha256,
                completed_epoch=epoch,
                actor=actor,
                critic=critic,
                rng=rng,
                best_actor=best_actor,
                best_critic=best_critic,
                history=history,
                best_key=best_key,
                best_epoch=best_epoch,
            )
    return best_actor, _clone_policy(actor), best_critic, critic.clone(), {
        "format_version": 1,
        "algorithm": "masked PPO with GAE and BC warm start",
        "seed": seed,
        "feature_names": list(actor.feature_names),
        "reward": "negative incremental makespan divided by HEFT makespan",
        "reward_sum_identity": "sum(step_reward) == -final_ratio",
        "epochs": history,
        "selection": {
            "split": "validation",
            "test_accessed": False,
            "metric": "validation_mean_ratio",
            "candidates": "BC warm start plus every PPO epoch",
            "tie_break": ["zero_failures", "lower_mean_ratio", "earlier_epoch"],
            "best_epoch": best_epoch,
        },
    }


def _checkpoint_metadata(
    actor_path: Path,
    actor: MaskedMLPPolicy,
    value_path: Path,
    critic: ValueNetwork,
) -> dict[str, Any]:
    return {
        "actor": {
            "name": actor_path.name,
            "sha256": _file_hash(actor_path),
            "parameter_sha256": policy_parameter_hash(actor),
        },
        "value": {
            "name": value_path.name,
            "sha256": _file_hash(value_path),
            "parameter_sha256": value_parameter_hash(critic),
        },
    }


def _task_gnn_checkpoint_metadata(
    actor_path: Path,
    actor: TaskGNNPolicy,
    value_path: Path,
    critic: ValueNetwork,
) -> dict[str, Any]:
    return {
        "actor": {
            "name": actor_path.name,
            "sha256": _file_hash(actor_path),
            "parameter_sha256": task_gnn_parameter_hash(actor),
            **task_gnn_metadata(actor),
        },
        "value": {
            "name": value_path.name,
            "sha256": _file_hash(value_path),
            "parameter_sha256": value_parameter_hash(critic),
        },
    }


def _read_seed_extension_json(path: Path, error_path: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail("ppo_seed_extension_read", error_path, str(error))
    if not isinstance(value, dict):
        _fail("ppo_seed_extension_read", error_path, "expected an object")
    return value


def _seed_extension_training_contract(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    contract = json.loads(json.dumps(config))
    contract.pop("seeds", None)
    contract.pop("output_dir", None)
    contract.pop("seed_extension", None)
    return contract


def _prepare_seed_extension(
    config_source: Path,
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    extension_config = config.get("seed_extension")
    if extension_config is None:
        return None

    source_dir = _resolve_config_path(
        config_source, extension_config["source_dir"]
    )
    if not source_dir.is_dir():
        _fail(
            "ppo_seed_extension_source",
            "$.seed_extension.source_dir",
            f"source evidence directory does not exist: {source_dir}",
        )
    source_manifest_path = source_dir / "ppo_run_manifest.json"
    if not source_manifest_path.is_file():
        _fail(
            "ppo_seed_extension_source",
            "$.seed_extension.source_dir",
            "source evidence directory has no ppo_run_manifest.json",
        )
    actual_manifest_sha256 = _file_hash(source_manifest_path)
    expected_manifest_sha256 = extension_config["run_manifest_sha256"]
    if actual_manifest_sha256 != expected_manifest_sha256:
        _fail(
            "ppo_seed_extension_manifest",
            "$.seed_extension.run_manifest_sha256",
            "source run manifest hash mismatch",
            details={
                "expected_sha256": expected_manifest_sha256,
                "actual_sha256": actual_manifest_sha256,
            },
        )

    source_manifest = _read_seed_extension_json(
        source_manifest_path,
        "$.seed_extension.source_run_manifest",
    )
    if (
        source_manifest.get("format_version") != 1
        or source_manifest.get("mode") != "stg_masked_ppo"
    ):
        _fail(
            "ppo_seed_extension_contract",
            "$.seed_extension.source_run_manifest",
            "source run must be a format-v1 masked-MLP PPO run",
        )

    artifacts = source_manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        _fail(
            "ppo_seed_extension_contract",
            "$.seed_extension.source_run_manifest.artifacts",
            "source artifact manifest must be an object",
        )
    for name, metadata in artifacts.items():
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or not isinstance(metadata, dict)
            or not isinstance(metadata.get("sha256"), str)
            or isinstance(metadata.get("bytes"), bool)
            or not isinstance(metadata.get("bytes"), int)
            or metadata["bytes"] < 0
        ):
            _fail(
                "ppo_seed_extension_contract",
                "$.seed_extension.source_run_manifest.artifacts",
                "source artifact metadata is malformed",
            )
        path = source_dir / name
        try:
            size = path.stat().st_size
        except OSError as error:
            _fail(
                "ppo_seed_extension_artifact",
                f"$.seed_extension.artifacts.{name}",
                str(error),
            )
        actual_hash = _file_hash(path)
        if size != metadata["bytes"] or actual_hash != metadata["sha256"]:
            _fail(
                "ppo_seed_extension_artifact",
                f"$.seed_extension.artifacts.{name}",
                "source artifact size/hash mismatch",
                details={
                    "expected_bytes": metadata["bytes"],
                    "actual_bytes": size,
                    "expected_sha256": metadata["sha256"],
                    "actual_sha256": actual_hash,
                },
            )

    required_control_artifacts = {"resolved_config.json", "ppo_summary.json"}
    missing_control_artifacts = sorted(
        required_control_artifacts - set(artifacts)
    )
    if missing_control_artifacts:
        _fail(
            "ppo_seed_extension_artifact",
            "$.seed_extension.source_run_manifest.artifacts",
            "source control artifact set is incomplete",
            details={"missing": missing_control_artifacts},
        )
    source_config = _read_seed_extension_json(
        source_dir / "resolved_config.json",
        "$.seed_extension.source_resolved_config",
    )
    source_summary = _read_seed_extension_json(
        source_dir / "ppo_summary.json",
        "$.seed_extension.source_summary",
    )
    reuse_seeds = list(extension_config["reuse_seeds"])
    if (
        source_config.get("seeds") != reuse_seeds
        or source_manifest.get("inputs", {}).get("seeds") != reuse_seeds
    ):
        _fail(
            "ppo_seed_extension_contract",
            "$.seed_extension.reuse_seeds",
            "source seeds do not exactly match reuse_seeds",
        )
    if _seed_extension_training_contract(
        source_config
    ) != _seed_extension_training_contract(config):
        _fail(
            "ppo_seed_extension_contract",
            "$.seed_extension.source_resolved_config",
            "source and target training contracts differ",
        )
    if source_manifest.get("inputs", {}).get("test_accessed") is not False:
        _fail(
            "ppo_seed_extension_contract",
            "$.seed_extension.source_run_manifest.inputs.test_accessed",
            "source run must declare test_accessed=false",
        )
    if source_summary.get("data_access", {}).get("test_accessed") is not False:
        _fail(
            "ppo_seed_extension_contract",
            "$.seed_extension.source_summary.data_access.test_accessed",
            "source summary must declare test_accessed=false",
        )

    raw_seed_results = source_summary.get("seeds")
    if not isinstance(raw_seed_results, list):
        _fail(
            "ppo_seed_extension_contract",
            "$.seed_extension.source_summary.seeds",
            "source summary seeds must be an array",
        )
    seed_results: dict[int, dict[str, Any]] = {}
    for result in raw_seed_results:
        if not isinstance(result, dict) or isinstance(result.get("seed"), bool):
            _fail(
                "ppo_seed_extension_contract",
                "$.seed_extension.source_summary.seeds",
                "source summary contains an invalid seed result",
            )
        seed = result.get("seed")
        if not isinstance(seed, int) or seed in seed_results:
            _fail(
                "ppo_seed_extension_contract",
                "$.seed_extension.source_summary.seeds",
                "source summary seed results must be unique integers",
            )
        seed_results[seed] = result
    if list(seed_results) != reuse_seeds:
        _fail(
            "ppo_seed_extension_contract",
            "$.seed_extension.source_summary.seeds",
            "source summary seed order does not match reuse_seeds",
        )
    return {
        "source_dir": source_dir,
        "source_manifest_path": source_manifest_path,
        "source_manifest_sha256": actual_manifest_sha256,
        "seed_results": seed_results,
        "artifacts": artifacts,
        "reuse_seeds": reuse_seeds,
        "new_seeds": [
            seed for seed in config["seeds"] if seed not in reuse_seeds
        ],
    }


def _inherited_seed_artifact_names(
    extension: Mapping[str, Any],
    seed: int,
) -> list[str]:
    prefix = f"seed_{seed}"
    names = [
        f"{prefix}_bc_warm_start.npz",
        f"{prefix}_ppo_best_policy.npz",
        f"{prefix}_ppo_best_value.npz",
        f"{prefix}_ppo_last_policy.npz",
        f"{prefix}_ppo_last_value.npz",
        f"{prefix}_training_curve.json",
        f"{prefix}_validation_diagnostics.json",
        f"{prefix}_validation_failures.jsonl",
    ]
    state_name = f"{prefix}_ppo_training_state.npz"
    if state_name in extension["artifacts"]:
        names.append(state_name)
    missing = [name for name in names if name not in extension["artifacts"]]
    if missing:
        _fail(
            "ppo_seed_extension_artifact",
            f"$.seed_extension.seeds.{seed}",
            "source seed artifact set is incomplete",
            details={"missing": missing},
        )
    return names


def _inherit_seed_result(
    extension: Mapping[str, Any],
    output_dir: Path,
    seed: int,
    *,
    resume: bool,
) -> tuple[dict[str, Any], list[str]]:
    names = _inherited_seed_artifact_names(extension, seed)
    for name in names:
        source = extension["source_dir"] / name
        destination = output_dir / name
        expected = extension["artifacts"][name]
        if resume:
            try:
                actual_size = destination.stat().st_size
            except OSError as error:
                _fail(
                    "ppo_seed_extension_artifact",
                    f"$.output_dir.{name}",
                    str(error),
                )
            actual_hash = _file_hash(destination)
            if (
                actual_size != expected["bytes"]
                or actual_hash != expected["sha256"]
            ):
                _fail(
                    "ppo_seed_extension_artifact",
                    f"$.output_dir.{name}",
                    "inherited output artifact differs from source",
                )
        else:
            shutil.copy2(source, destination)
    result = json.loads(json.dumps(extension["seed_results"][seed]))
    if "training_state" not in result:
        result["training_state"] = None
    result["seed_extension"] = {
        "mode": "inherited_verified",
        "source_run_manifest_sha256": extension[
            "source_manifest_sha256"
        ],
    }
    return result, names


def _run_ppo_pipeline_in_directory(
    config_path: str | Path,
    output_override: str | Path | None = None,
    *,
    resume: bool = False,
) -> Path:
    """Run the PPO pipeline against its final or transaction staging directory."""

    config_source = Path(config_path).resolve()
    config = load_ppo_config(config_source)
    seed_extension = _prepare_seed_extension(config_source, config)
    manifest_path = _resolve_config_path(
        config_source, config["benchmark"]["manifest"]
    )
    raw_root = _resolve_config_path(config_source, config["benchmark"]["raw_root"])
    output_dir = (
        _resolve_config_path(config_source, config["output_dir"])
        if output_override is None
        else Path(output_override).resolve()
    )
    if seed_extension is not None and seed_extension["source_dir"] == output_dir:
        _fail(
            "ppo_seed_extension_source",
            "$.seed_extension.source_dir",
            "source and target output directories must differ",
        )
    if resume:
        if not output_dir.is_dir() or not any(output_dir.iterdir()):
            _fail(
                "ppo_resume_output_missing",
                "$.output_dir",
                f"resume output does not exist or is empty: {output_dir}",
            )
        resolved_path = output_dir / "resolved_config.json"
        try:
            previous_config = json.loads(resolved_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            _fail(
                "ppo_resume_config_read",
                "$.output_dir.resolved_config",
                str(error),
            )
        if previous_config != config:
            _fail(
                "ppo_resume_config_mismatch",
                "$.output_dir.resolved_config",
                "current normalized config differs from the interrupted run",
                details={
                    "previous_sha256": _canonical_sha256(previous_config),
                    "current_sha256": _canonical_sha256(config),
                },
            )
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            _fail(
                "output_not_empty",
                "$.output_dir",
                f"refusing to mix evidence in {output_dir}",
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "resolved_config.json", config)

    print("[1/6] loading verified train and validation splits (test forbidden)")
    benchmark_manifest = load_benchmark_manifest(manifest_path)
    manifest_sha256 = _file_hash(manifest_path)
    train_scenarios = load_frozen_split(
        raw_root,
        manifest_path,
        "train",
        purpose="teacher",
    )
    validation_scenarios = load_frozen_split(
        raw_root,
        manifest_path,
        "validation",
        purpose="model_selection",
    )
    train_entries = [
        entry for entry in benchmark_manifest["entries"] if entry["split"] == "train"
    ]
    validation_entries = [
        entry
        for entry in benchmark_manifest["entries"]
        if entry["split"] == "validation"
    ]
    repository = Path(__file__).resolve().parents[1]
    code_metadata = _git_metadata(repository)
    code_source_sha256 = {
        path.relative_to(repository).as_posix(): _file_hash(path)
        for path in sorted((repository / "trisched").glob("*.py"))
    }

    print("[2/6] generating production/independent HEFT manifests")
    train_teacher = build_teacher_manifest(
        train_scenarios,
        train_entries,
        split="train",
        purpose="behavior_cloning_teacher",
        benchmark_manifest_name=manifest_path.name,
        benchmark_manifest_sha256=manifest_sha256,
        benchmark_id=benchmark_manifest["benchmark_id"],
        code_metadata=code_metadata,
    )
    validation_reference = build_teacher_manifest(
        validation_scenarios,
        validation_entries,
        split="validation",
        purpose="model_selection_reference",
        benchmark_manifest_name=manifest_path.name,
        benchmark_manifest_sha256=manifest_sha256,
        benchmark_id=benchmark_manifest["benchmark_id"],
        code_metadata=code_metadata,
    )
    teacher_artifacts = {
        "train_teacher_manifest.json": train_teacher,
        "validation_reference_manifest.json": validation_reference,
    }
    if resume:
        for name, expected in teacher_artifacts.items():
            path = output_dir / name
            try:
                actual = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                _fail(
                    "ppo_resume_artifact_read",
                    f"$.output_dir.{name}",
                    str(error),
                )
            if actual != expected:
                _fail(
                    "ppo_resume_artifact_mismatch",
                    f"$.output_dir.{name}",
                    "resume provenance artifact differs from the current run",
                    details={
                        "previous_sha256": _canonical_sha256(actual),
                        "current_sha256": _canonical_sha256(expected),
                    },
                )
        failure_path = output_dir / "teacher_failures.jsonl"
        try:
            failure_bytes = failure_path.read_bytes()
        except OSError as error:
            _fail(
                "ppo_resume_artifact_read",
                "$.output_dir.teacher_failures.jsonl",
                str(error),
            )
        if failure_bytes:
            _fail(
                "ppo_resume_artifact_mismatch",
                "$.output_dir.teacher_failures.jsonl",
                "teacher failure evidence must remain empty",
            )
    else:
        for name, value in teacher_artifacts.items():
            _write_json(output_dir / name, value)
        _write_jsonl(output_dir / "teacher_failures.jsonl", [])

    print("[3/6] freezing teacher states and running feature ablation reference")
    train_states = freeze_teacher_dataset(
        train_scenarios,
        train_teacher,
        split="train",
        purpose="behavior_cloning_teacher",
    )
    validation_states = freeze_teacher_dataset(
        validation_scenarios,
        validation_reference,
        split="validation",
        purpose="model_selection_reference",
    )
    bc_common = {
        **config["behavior_cloning"],
        "failure_penalty_ratio": config["selection"]["failure_penalty_ratio"],
    }
    reference_seed = int(config["ablation"]["teacher_feature_reference_seed"])
    reference_actor, _, reference_curve = train_bc_baseline(
        train_scenarios,
        train_teacher,
        validation_scenarios,
        validation_reference,
        {**bc_common, "feature_names": list(FEATURE_NAMES)},
        seed=reference_seed,
        train_frozen_states=train_states,
        validation_frozen_states=validation_states,
    )
    reference_metrics, _ = evaluate_bc_policy(
        reference_actor,
        validation_scenarios,
        validation_reference,
        failure_penalty_ratio=config["selection"]["failure_penalty_ratio"],
        frozen_teacher_states=validation_states,
    )
    reference_path = output_dir / "ablation_with_teacher_bc.npz"
    reference_actor.save(reference_path)

    print("[4/6] training no-teacher-feature BC plus masked PPO across seeds")
    artifact_names = [
        "resolved_config.json",
        "train_teacher_manifest.json",
        "validation_reference_manifest.json",
        "teacher_failures.jsonl",
        reference_path.name,
    ]
    extension_manifest_snapshot: str | None = None
    if seed_extension is not None:
        extension_manifest_snapshot = (
            "seed_extension_source_run_manifest.json"
        )
        snapshot_path = output_dir / extension_manifest_snapshot
        if resume:
            if (
                not snapshot_path.is_file()
                or _file_hash(snapshot_path)
                != seed_extension["source_manifest_sha256"]
            ):
                _fail(
                    "ppo_seed_extension_artifact",
                    f"$.output_dir.{extension_manifest_snapshot}",
                    "source run manifest snapshot is missing or changed",
                )
        else:
            shutil.copy2(
                seed_extension["source_manifest_path"],
                snapshot_path,
            )
        artifact_names.append(extension_manifest_snapshot)
    seed_results: list[dict[str, Any]] = []
    resumed_seeds: list[int] = []
    first_no_teacher_metrics: dict[str, Any] | None = None
    for seed in config["seeds"]:
        if (
            seed_extension is not None
            and seed in seed_extension["reuse_seeds"]
        ):
            inherited_result, inherited_names = _inherit_seed_result(
                seed_extension,
                output_dir,
                int(seed),
                resume=resume,
            )
            if int(seed) == reference_seed:
                first_no_teacher_metrics = inherited_result[
                    "warm_start_validation"
                ]
            seed_results.append(inherited_result)
            artifact_names.extend(inherited_names)
            continue
        prefix = f"seed_{seed}"
        warm_actor, _, warm_curve = train_bc_baseline(
            train_scenarios,
            train_teacher,
            validation_scenarios,
            validation_reference,
            {
                **bc_common,
                "feature_names": config["features"]["selected"],
            },
            seed=int(seed),
            train_frozen_states=train_states,
            validation_frozen_states=validation_states,
        )
        warm_metrics, _ = evaluate_bc_policy(
            warm_actor,
            validation_scenarios,
            validation_reference,
            failure_penalty_ratio=config["selection"]["failure_penalty_ratio"],
            frozen_teacher_states=validation_states,
        )
        if int(seed) == reference_seed:
            first_no_teacher_metrics = warm_metrics
        warm_path = output_dir / f"{prefix}_bc_warm_start.npz"
        warm_actor.save(warm_path)
        resume_state_path = output_dir / f"{prefix}_ppo_training_state.npz"
        state_exists = resume_state_path.is_file()
        completed_output_names = [
            f"{prefix}_ppo_best_policy.npz",
            f"{prefix}_ppo_last_policy.npz",
            f"{prefix}_ppo_best_value.npz",
            f"{prefix}_ppo_last_value.npz",
            f"{prefix}_training_curve.json",
            f"{prefix}_validation_diagnostics.json",
            f"{prefix}_validation_failures.jsonl",
        ]
        if resume and not state_exists and any(
            (output_dir / name).exists() for name in completed_output_names
        ):
            _fail(
                "ppo_resume_state_missing",
                "$.resume",
                f"seed {seed} has PPO outputs but no training state",
            )
        resume_contract = {
            "format_version": 1,
            "algorithm": "masked_ppo_epoch_boundary_resume",
            "seed": int(seed),
            "config_source_sha256": _file_hash(config_source),
            "benchmark_manifest_sha256": manifest_sha256,
            "code": {
                "git": code_metadata,
                "sources": code_source_sha256,
            },
            "features": config["features"]["selected"],
            "ppo": {
                **config["ppo"],
                "failure_penalty_ratio": config["selection"][
                    "failure_penalty_ratio"
                ],
            },
            "warm_start_parameter_sha256": policy_parameter_hash(warm_actor),
            "train": {
                "scenario_hashes": [
                    scenario.content_hash() for scenario in train_scenarios
                ],
                "teacher_trace_hashes_sha256": train_teacher[
                    "trace_hashes_sha256"
                ],
                "teacher_schedule_hashes_sha256": train_teacher[
                    "schedule_hashes_sha256"
                ],
            },
            "validation": {
                "scenario_hashes": [
                    scenario.content_hash() for scenario in validation_scenarios
                ],
                "reference_trace_hashes_sha256": validation_reference[
                    "trace_hashes_sha256"
                ],
                "reference_schedule_hashes_sha256": validation_reference[
                    "schedule_hashes_sha256"
                ],
            },
            "test_accessed": False,
        }
        resume_seed = resume and state_exists
        if resume_seed:
            resumed_seeds.append(int(seed))
        (
            best_actor,
            last_actor,
            best_value,
            last_value,
            ppo_curve,
        ) = train_masked_ppo(
            warm_actor,
            train_scenarios,
            train_teacher,
            validation_scenarios,
            validation_reference,
            {
                **config["ppo"],
                "failure_penalty_ratio": config["selection"][
                    "failure_penalty_ratio"
                ],
            },
            seed=int(seed),
            validation_frozen_states=validation_states,
            resume_state_path=resume_state_path,
            resume_contract=resume_contract,
            resume=resume_seed,
        )
        best_metrics, best_rows = evaluate_bc_policy(
            best_actor,
            validation_scenarios,
            validation_reference,
            failure_penalty_ratio=config["selection"]["failure_penalty_ratio"],
            frozen_teacher_states=validation_states,
        )
        last_metrics, last_rows = evaluate_bc_policy(
            last_actor,
            validation_scenarios,
            validation_reference,
            failure_penalty_ratio=config["selection"]["failure_penalty_ratio"],
            frozen_teacher_states=validation_states,
        )
        best_actor_path = output_dir / f"{prefix}_ppo_best_policy.npz"
        last_actor_path = output_dir / f"{prefix}_ppo_last_policy.npz"
        best_value_path = output_dir / f"{prefix}_ppo_best_value.npz"
        last_value_path = output_dir / f"{prefix}_ppo_last_value.npz"
        best_actor.save(best_actor_path)
        last_actor.save(last_actor_path)
        best_value.save(best_value_path)
        last_value.save(last_value_path)
        curve_path = output_dir / f"{prefix}_training_curve.json"
        diagnostics_path = output_dir / f"{prefix}_validation_diagnostics.json"
        failures_path = output_dir / f"{prefix}_validation_failures.jsonl"
        _write_json(
            curve_path,
            {
                "format_version": 1,
                "seed": seed,
                "behavior_cloning": warm_curve,
                "ppo": ppo_curve,
            },
        )
        _write_json(
            diagnostics_path,
            {
                "format_version": 1,
                "seed": seed,
                "split": "validation",
                "test_accessed": False,
                "best": {"metrics": best_metrics, "per_instance": best_rows},
                "last": {"metrics": last_metrics, "per_instance": last_rows},
            },
        )
        _write_jsonl(
            failures_path,
            [row for row in best_rows if row["status"] == "failure"],
        )
        best_checkpoint = _checkpoint_metadata(
            best_actor_path,
            best_actor,
            best_value_path,
            best_value,
        )
        last_checkpoint = _checkpoint_metadata(
            last_actor_path,
            last_actor,
            last_value_path,
            last_value,
        )
        training_state = {
            "name": resume_state_path.name,
            "sha256": _file_hash(resume_state_path),
            "format_version": 1,
            "completed_epoch": int(config["ppo"]["epochs"]),
            "boundary": "completed_ppo_epoch",
        }
        seed_result = {
            "seed": seed,
            "warm_start_validation": warm_metrics,
            "selection": ppo_curve["selection"],
            "best_validation": best_metrics,
            "last_validation": last_metrics,
            "best_checkpoint": best_checkpoint,
            "last_checkpoint": last_checkpoint,
            "training_state": training_state,
            "improved_over_warm_start": best_metrics["mean_ratio"]
            < warm_metrics["mean_ratio"] - 1e-12,
            "selected_warm_start": ppo_curve["selection"]["best_epoch"] == 0,
        }
        if seed_extension is not None:
            seed_result["seed_extension"] = {
                "mode": "trained_current_run"
            }
        seed_results.append(seed_result)
        artifact_names.extend(
            [
                warm_path.name,
                best_actor_path.name,
                last_actor_path.name,
                best_value_path.name,
                last_value_path.name,
                curve_path.name,
                diagnostics_path.name,
                failures_path.name,
                resume_state_path.name,
            ]
        )
    assert first_no_teacher_metrics is not None

    print("[5/6] aggregating validation gate and writing summary")
    ablation = {
        "format_version": 1,
        "seed": reference_seed,
        "controlled_change": "remove is_heft_task and is_heft_pair",
        "with_teacher_features": {
            "feature_names": list(FEATURE_NAMES),
            "training": reference_curve,
            "validation": reference_metrics,
            "checkpoint": {
                "name": reference_path.name,
                "sha256": _file_hash(reference_path),
                "parameter_sha256": policy_parameter_hash(reference_actor),
            },
        },
        "without_teacher_features": {
            "feature_names": config["features"]["selected"],
            "validation": first_no_teacher_metrics,
        },
        "test_accessed": False,
    }
    ablation_path = output_dir / "feature_ablation.json"
    _write_json(ablation_path, ablation)
    artifact_names.append(ablation_path.name)

    seed_ratios = [
        float(result["best_validation"]["mean_ratio"])
        for result in seed_results
    ]
    validation_gate_passed = all(
        result["best_validation"]["failure_count"] == 0
        and result["best_validation"]["illegal_action_count"] == 0
        and result["best_validation"]["mean_ratio"]
        <= float(config["selection"]["target_ratio"]) + 1e-12
        for result in seed_results
    )
    summary = {
        "format_version": 1,
        "mode": "stg_masked_ppo",
        "benchmark_id": benchmark_manifest["benchmark_id"],
        "data_access": {
            "loaded_splits": ["train", "validation"],
            "teacher_split": "train",
            "training_split": "train",
            "model_selection_split": "validation",
            "test_accessed": False,
            "test_status": "forbidden_until_final_evaluation",
        },
        "features": config["features"],
        "teacher": {
            "scenario_count": train_teacher["scenario_count"],
            "action_count": train_teacher["action_count"],
            "failure_count": train_teacher["failure_count"],
            "trace_hashes_sha256": train_teacher["trace_hashes_sha256"],
            "schedule_hashes_sha256": train_teacher["schedule_hashes_sha256"],
        },
        "ablation": "feature_ablation.json",
        "seeds": seed_results,
        "aggregate_validation": {
            "seed_count": len(seed_results),
            "mean_of_seed_mean_ratios": float(np.mean(seed_ratios)),
            "std_of_seed_mean_ratios": float(np.std(seed_ratios)),
            "min_seed_mean_ratio": float(np.min(seed_ratios)),
            "max_seed_mean_ratio": float(np.max(seed_ratios)),
            "improved_seed_count": sum(
                int(result["improved_over_warm_start"])
                for result in seed_results
            ),
            "warm_start_fallback_count": sum(
                int(result["selected_warm_start"]) for result in seed_results
            ),
            "failure_count": sum(
                int(result["best_validation"]["failure_count"])
                for result in seed_results
            ),
            "illegal_action_count": sum(
                int(result["best_validation"]["illegal_action_count"])
                for result in seed_results
            ),
        },
        "selection": {
            **config["selection"],
            "split": "validation",
            "test_accessed": False,
        },
        "resumability": {
            "state_format_version": 1,
            "boundary": "completed_ppo_epoch",
            "state_artifacts": [
                result["training_state"]["name"]
                for result in seed_results
                if result.get("training_state") is not None
            ],
            "test_accessed": False,
        },
        "validation_gate_passed": validation_gate_passed,
        "recommendation": (
            "proceed_to_independent_review"
            if validation_gate_passed
            else "fallback_to_p1_a01_and_tune_only_on_validation"
        ),
        "run_manifest": "ppo_run_manifest.json",
    }
    if seed_extension is not None:
        summary["seed_extension"] = {
            "source_run_manifest": extension_manifest_snapshot,
            "source_run_manifest_sha256": seed_extension[
                "source_manifest_sha256"
            ],
            "inherited_seeds": seed_extension["reuse_seeds"],
            "trained_seeds": seed_extension["new_seeds"],
            "source_test_accessed": False,
        }
    summary_path = output_dir / "ppo_summary.json"
    _write_json(summary_path, summary)
    artifact_names.append(summary_path.name)

    print("[6/6] writing reproducibility manifest")
    artifacts = {
        name: {
            "bytes": (output_dir / name).stat().st_size,
            "sha256": _file_hash(output_dir / name),
        }
        for name in sorted(artifact_names)
    }
    lockfile = repository / "requirements-lock.txt"
    execution = {
        "resume_requested": resume,
        "resumed_seeds": resumed_seeds,
        "resume_boundary": "completed_ppo_epoch",
        "publication_mode": (
            "staging_directory_swap" if resume else "direct_new_directory"
        ),
    }
    if seed_extension is not None:
        execution["seed_extension"] = {
            "source_run_manifest": extension_manifest_snapshot,
            "source_run_manifest_sha256": seed_extension[
                "source_manifest_sha256"
            ],
            "inherited_seeds": seed_extension["reuse_seeds"],
            "trained_seeds": seed_extension["new_seeds"],
        }
    run_manifest = {
        "format_version": 1,
        "mode": "stg_masked_ppo",
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "code": code_metadata,
        "runtime": {
            "trisched": __version__,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "execution": execution,
        "inputs": {
            "config": {
                "name": config_source.name,
                "sha256": _file_hash(config_source),
            },
            "benchmark_manifest": {
                "name": manifest_path.name,
                "sha256": manifest_sha256,
            },
            "dependency_lock": {
                "name": lockfile.name,
                "sha256": _file_hash(lockfile),
            }
            if lockfile.is_file()
            else None,
            "splits": {
                "train": benchmark_manifest["splits"]["train"],
                "validation": benchmark_manifest["splits"]["validation"],
            },
            "feature_names": config["features"]["selected"],
            "seeds": config["seeds"],
            "test_accessed": False,
        },
        "checkpoints": {
            str(result["seed"]): {
                "best": result["best_checkpoint"],
                "last": result["last_checkpoint"],
                "training_state": result.get("training_state"),
                **(
                    {"seed_extension": result["seed_extension"]}
                    if "seed_extension" in result
                    else {}
                ),
            }
            for result in seed_results
        },
        "artifacts": artifacts,
    }
    _write_json(output_dir / "ppo_run_manifest.json", run_manifest)
    print(
        "done: seed mean ratios="
        + ", ".join(f"{ratio:.6f}" for ratio in seed_ratios)
        + f", validation_gate_passed={validation_gate_passed}"
    )
    print(f"summary: {summary_path.resolve()}")
    return summary_path


def _resume_transaction_path(output_dir: Path, role: str) -> Path:
    return output_dir.with_name(
        f".{output_dir.name}.ppo-resume-{role}-{uuid.uuid4().hex}"
    )


def _remove_resume_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _publish_resume_staging(staging_dir: Path, output_dir: Path) -> None:
    backup_dir = _resume_transaction_path(output_dir, "backup")
    try:
        os.replace(output_dir, backup_dir)
    except OSError as error:
        try:
            _remove_resume_tree(staging_dir)
        except OSError:
            pass
        _fail(
            "ppo_resume_publish",
            "$.output_dir",
            f"could not move the previous evidence directory aside: {error}",
        )
    try:
        os.replace(staging_dir, output_dir)
    except OSError as error:
        rollback_error: OSError | None = None
        try:
            os.replace(backup_dir, output_dir)
        except OSError as captured:
            rollback_error = captured
        try:
            _remove_resume_tree(staging_dir)
        except OSError:
            pass
        _fail(
            "ppo_resume_publish",
            "$.output_dir",
            f"could not publish the completed resume transaction: {error}",
            details={
                "rollback_error": str(rollback_error)
                if rollback_error is not None
                else None,
                "backup_dir": str(backup_dir) if rollback_error is not None else None,
            },
        )
    try:
        _remove_resume_tree(backup_dir)
    except OSError as error:
        _fail(
            "ppo_resume_publish",
            "$.output_dir",
            f"published resume output but could not remove its backup: {error}",
            details={"backup_dir": str(backup_dir)},
        )


def run_ppo_pipeline(
    config_path: str | Path,
    output_override: str | Path | None = None,
    *,
    resume: bool = False,
) -> Path:
    """Run the no-test multi-seed PPO pipeline with transactional resume."""

    if not resume:
        return _run_ppo_pipeline_in_directory(
            config_path,
            output_override,
            resume=False,
        )

    config_source = Path(config_path).resolve()
    config = load_ppo_config(config_source)
    output_dir = (
        _resolve_config_path(config_source, config["output_dir"])
        if output_override is None
        else Path(output_override).resolve()
    )
    if not output_dir.is_dir() or not any(output_dir.iterdir()):
        _fail(
            "ppo_resume_output_missing",
            "$.output_dir",
            f"resume output does not exist or is empty: {output_dir}",
        )

    staging_dir = _resume_transaction_path(output_dir, "staging")
    try:
        shutil.copytree(output_dir, staging_dir)
    except (OSError, shutil.Error) as error:
        try:
            _remove_resume_tree(staging_dir)
        except OSError:
            pass
        _fail(
            "ppo_resume_stage_write",
            "$.output_dir",
            f"could not create resume transaction staging: {error}",
        )

    try:
        staged_summary = _run_ppo_pipeline_in_directory(
            config_source,
            staging_dir,
            resume=True,
        )
    except BaseException:
        try:
            _remove_resume_tree(staging_dir)
        except OSError:
            pass
        raise

    _publish_resume_staging(staging_dir, output_dir)
    return output_dir / staged_summary.name


def _run_task_gnn_pipeline_in_directory(
    config_path: str | Path,
    output_override: str | Path | None = None,
    *,
    resume: bool = False,
) -> Path:
    """Run P1-A03 against a new directory or a copied resume staging tree."""

    config_source = Path(config_path).resolve()
    config = load_task_gnn_config(config_source)
    manifest_path = _resolve_config_path(
        config_source, config["benchmark"]["manifest"]
    )
    raw_root = _resolve_config_path(config_source, config["benchmark"]["raw_root"])
    output_dir = (
        _resolve_config_path(config_source, config["output_dir"])
        if output_override is None
        else Path(output_override).resolve()
    )
    if resume:
        if not output_dir.is_dir() or not any(output_dir.iterdir()):
            _fail(
                "task_gnn_resume_output_missing",
                "$.output_dir",
                f"resume output does not exist or is empty: {output_dir}",
            )
        resolved_path = output_dir / "resolved_config.json"
        try:
            previous_config = json.loads(resolved_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            _fail(
                "task_gnn_resume_config_read",
                "$.output_dir.resolved_config",
                str(error),
            )
        if previous_config != config:
            _fail(
                "task_gnn_resume_config_mismatch",
                "$.output_dir.resolved_config",
                "current normalized config differs from the interrupted run",
                details={
                    "previous_sha256": _canonical_sha256(previous_config),
                    "current_sha256": _canonical_sha256(config),
                },
            )
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            _fail(
                "output_not_empty",
                "$.output_dir",
                f"refusing to mix evidence in {output_dir}",
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "resolved_config.json", config)

    print("[1/6] loading verified task-GNN train and validation splits")
    benchmark_manifest = load_benchmark_manifest(manifest_path)
    manifest_sha256 = _file_hash(manifest_path)
    train_scenarios = load_frozen_split(
        raw_root,
        manifest_path,
        "train",
        purpose="teacher",
    )
    validation_scenarios = load_frozen_split(
        raw_root,
        manifest_path,
        "validation",
        purpose="model_selection",
    )
    train_entries = [
        entry for entry in benchmark_manifest["entries"] if entry["split"] == "train"
    ]
    validation_entries = [
        entry
        for entry in benchmark_manifest["entries"]
        if entry["split"] == "validation"
    ]
    repository = Path(__file__).resolve().parents[1]
    code_metadata = _git_metadata(repository)
    code_source_sha256 = {
        path.relative_to(repository).as_posix(): _file_hash(path)
        for path in sorted((repository / "trisched").glob("*.py"))
    }

    print("[2/6] generating production/independent HEFT manifests")
    train_teacher = build_teacher_manifest(
        train_scenarios,
        train_entries,
        split="train",
        purpose="behavior_cloning_teacher",
        benchmark_manifest_name=manifest_path.name,
        benchmark_manifest_sha256=manifest_sha256,
        benchmark_id=benchmark_manifest["benchmark_id"],
        code_metadata=code_metadata,
    )
    validation_reference = build_teacher_manifest(
        validation_scenarios,
        validation_entries,
        split="validation",
        purpose="model_selection_reference",
        benchmark_manifest_name=manifest_path.name,
        benchmark_manifest_sha256=manifest_sha256,
        benchmark_id=benchmark_manifest["benchmark_id"],
        code_metadata=code_metadata,
    )
    teacher_artifacts = {
        "train_teacher_manifest.json": train_teacher,
        "validation_reference_manifest.json": validation_reference,
    }
    if resume:
        for name, expected in teacher_artifacts.items():
            path = output_dir / name
            try:
                actual = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                _fail(
                    "task_gnn_resume_artifact_read",
                    f"$.output_dir.{name}",
                    str(error),
                )
            if actual != expected:
                _fail(
                    "task_gnn_resume_artifact_mismatch",
                    f"$.output_dir.{name}",
                    "resume provenance artifact differs from the current run",
                    details={
                        "previous_sha256": _canonical_sha256(actual),
                        "current_sha256": _canonical_sha256(expected),
                    },
                )
        failure_path = output_dir / "teacher_failures.jsonl"
        try:
            failure_bytes = failure_path.read_bytes()
        except OSError as error:
            _fail(
                "task_gnn_resume_artifact_read",
                "$.output_dir.teacher_failures.jsonl",
                str(error),
            )
        if failure_bytes:
            _fail(
                "task_gnn_resume_artifact_mismatch",
                "$.output_dir.teacher_failures.jsonl",
                "teacher failure evidence must remain empty",
            )
    else:
        for name, value in teacher_artifacts.items():
            _write_json(output_dir / name, value)
        _write_jsonl(output_dir / "teacher_failures.jsonl", [])

    print("[3/6] freezing the canonical 14-D task-GNN states")
    train_states = freeze_task_gnn_teacher_dataset(
        train_scenarios,
        train_teacher,
        split="train",
        purpose="behavior_cloning_teacher",
    )
    validation_states = freeze_task_gnn_teacher_dataset(
        validation_scenarios,
        validation_reference,
        split="validation",
        purpose="model_selection_reference",
    )
    bc_common = {
        **config["behavior_cloning"],
        "message_dim": config["task_gnn"]["message_dim"],
        "failure_penalty_ratio": config["selection"]["failure_penalty_ratio"],
    }

    print("[4/6] training task-GNN BC warm starts plus PPO across seeds")
    artifact_names = [
        "resolved_config.json",
        "train_teacher_manifest.json",
        "validation_reference_manifest.json",
        "teacher_failures.jsonl",
    ]
    seed_results: list[dict[str, Any]] = []
    resumed_seeds: list[int] = []
    architecture_metadata: dict[str, Any] | None = None
    for seed in config["seeds"]:
        prefix = f"seed_{seed}_task_gnn"
        warm_actor, _, warm_curve = train_task_gnn_bc_baseline(
            train_scenarios,
            train_teacher,
            validation_scenarios,
            validation_reference,
            bc_common,
            seed=int(seed),
            train_frozen_states=train_states,
            validation_frozen_states=validation_states,
        )
        architecture_metadata = task_gnn_metadata(warm_actor)
        warm_metrics, _ = evaluate_bc_policy(
            warm_actor,
            validation_scenarios,
            validation_reference,
            failure_penalty_ratio=config["selection"]["failure_penalty_ratio"],
            frozen_task_gnn_states=validation_states,
        )
        warm_path = output_dir / f"{prefix}_bc_warm_start.npz"
        warm_actor.save(warm_path)
        resume_state_path = output_dir / f"{prefix}_ppo_training_state.npz"
        state_exists = resume_state_path.is_file()
        completed_output_names = [
            f"{prefix}_ppo_best_policy.npz",
            f"{prefix}_ppo_last_policy.npz",
            f"{prefix}_ppo_best_value.npz",
            f"{prefix}_ppo_last_value.npz",
            f"{prefix}_training_curve.json",
            f"{prefix}_validation_diagnostics.json",
            f"{prefix}_validation_failures.jsonl",
        ]
        if resume and not state_exists and any(
            (output_dir / name).exists() for name in completed_output_names
        ):
            _fail(
                "ppo_resume_state_missing",
                "$.resume",
                f"seed {seed} has task-GNN PPO outputs but no training state",
            )
        resume_contract = {
            "format_version": 1,
            "algorithm": "task_gnn_masked_ppo_epoch_boundary_resume",
            "seed": int(seed),
            "normalized_config": config,
            "config_source_sha256": _file_hash(config_source),
            "benchmark_manifest_sha256": manifest_sha256,
            "code": {
                "git": code_metadata,
                "sources": code_source_sha256,
            },
            "architecture": architecture_metadata,
            "features": config["features"]["selected"],
            "ppo": {
                **config["ppo"],
                "failure_penalty_ratio": config["selection"][
                    "failure_penalty_ratio"
                ],
            },
            "warm_start_parameter_sha256": task_gnn_parameter_hash(warm_actor),
            "train": {
                "scenario_hashes": [
                    scenario.content_hash() for scenario in train_scenarios
                ],
                "teacher_trace_hashes_sha256": train_teacher[
                    "trace_hashes_sha256"
                ],
                "teacher_schedule_hashes_sha256": train_teacher[
                    "schedule_hashes_sha256"
                ],
            },
            "validation": {
                "scenario_hashes": [
                    scenario.content_hash() for scenario in validation_scenarios
                ],
                "reference_trace_hashes_sha256": validation_reference[
                    "trace_hashes_sha256"
                ],
                "reference_schedule_hashes_sha256": validation_reference[
                    "schedule_hashes_sha256"
                ],
            },
            "test_accessed": False,
        }
        resume_seed = resume and state_exists
        if resume_seed:
            resumed_seeds.append(int(seed))
        (
            best_actor,
            last_actor,
            best_value,
            last_value,
            ppo_curve,
        ) = train_task_gnn_ppo(
            warm_actor,
            train_scenarios,
            train_teacher,
            validation_scenarios,
            validation_reference,
            {
                **config["ppo"],
                "failure_penalty_ratio": config["selection"][
                    "failure_penalty_ratio"
                ],
            },
            seed=int(seed),
            validation_frozen_states=validation_states,
            resume_state_path=resume_state_path,
            resume_contract=resume_contract,
            resume=resume_seed,
        )
        best_metrics, best_rows = evaluate_bc_policy(
            best_actor,
            validation_scenarios,
            validation_reference,
            failure_penalty_ratio=config["selection"]["failure_penalty_ratio"],
            frozen_task_gnn_states=validation_states,
        )
        last_metrics, last_rows = evaluate_bc_policy(
            last_actor,
            validation_scenarios,
            validation_reference,
            failure_penalty_ratio=config["selection"]["failure_penalty_ratio"],
            frozen_task_gnn_states=validation_states,
        )
        best_actor_path = output_dir / f"{prefix}_ppo_best_policy.npz"
        last_actor_path = output_dir / f"{prefix}_ppo_last_policy.npz"
        best_value_path = output_dir / f"{prefix}_ppo_best_value.npz"
        last_value_path = output_dir / f"{prefix}_ppo_last_value.npz"
        best_actor.save(best_actor_path)
        last_actor.save(last_actor_path)
        best_value.save(best_value_path)
        last_value.save(last_value_path)
        curve_path = output_dir / f"{prefix}_training_curve.json"
        diagnostics_path = output_dir / f"{prefix}_validation_diagnostics.json"
        failures_path = output_dir / f"{prefix}_validation_failures.jsonl"
        _write_json(
            curve_path,
            {
                "format_version": 1,
                "seed": seed,
                "architecture": architecture_metadata,
                "behavior_cloning": warm_curve,
                "ppo": ppo_curve,
            },
        )
        _write_json(
            diagnostics_path,
            {
                "format_version": 1,
                "seed": seed,
                "split": "validation",
                "test_accessed": False,
                "best": {"metrics": best_metrics, "per_instance": best_rows},
                "last": {"metrics": last_metrics, "per_instance": last_rows},
            },
        )
        _write_jsonl(
            failures_path,
            [row for row in best_rows if row["status"] == "failure"],
        )
        best_checkpoint = _task_gnn_checkpoint_metadata(
            best_actor_path,
            best_actor,
            best_value_path,
            best_value,
        )
        last_checkpoint = _task_gnn_checkpoint_metadata(
            last_actor_path,
            last_actor,
            last_value_path,
            last_value,
        )
        training_state = {
            "name": resume_state_path.name,
            "sha256": _file_hash(resume_state_path),
            "format_version": 1,
            "completed_epoch": int(config["ppo"]["epochs"]),
            "boundary": "completed_ppo_epoch",
        }
        seed_results.append(
            {
                "seed": seed,
                "warm_start_validation": warm_metrics,
                "selection": ppo_curve["selection"],
                "best_validation": best_metrics,
                "last_validation": last_metrics,
                "best_checkpoint": best_checkpoint,
                "last_checkpoint": last_checkpoint,
                "training_state": training_state,
                "improved_over_warm_start": best_metrics["mean_ratio"]
                < warm_metrics["mean_ratio"] - 1e-12,
                "selected_warm_start": ppo_curve["selection"]["best_epoch"] == 0,
            }
        )
        artifact_names.extend(
            [
                warm_path.name,
                best_actor_path.name,
                last_actor_path.name,
                best_value_path.name,
                last_value_path.name,
                curve_path.name,
                diagnostics_path.name,
                failures_path.name,
                resume_state_path.name,
            ]
        )
    assert architecture_metadata is not None

    print("[5/6] aggregating the task-GNN validation gate")
    seed_ratios = [
        float(result["best_validation"]["mean_ratio"])
        for result in seed_results
    ]
    validation_gate_passed = all(
        result["best_validation"]["failure_count"] == 0
        and result["best_validation"]["illegal_action_count"] == 0
        and result["best_validation"]["mean_ratio"]
        <= float(config["selection"]["target_ratio"]) + 1e-12
        for result in seed_results
    )
    summary = {
        "format_version": 1,
        "mode": "stg_task_gnn_ppo",
        "benchmark_id": benchmark_manifest["benchmark_id"],
        "data_access": {
            "loaded_splits": ["train", "validation"],
            "teacher_split": "train",
            "training_split": "train",
            "model_selection_split": "validation",
            "test_accessed": False,
            "test_status": "forbidden_until_final_evaluation",
        },
        "architecture": architecture_metadata,
        "features": config["features"],
        "teacher": {
            "scenario_count": train_teacher["scenario_count"],
            "action_count": train_teacher["action_count"],
            "failure_count": train_teacher["failure_count"],
            "trace_hashes_sha256": train_teacher["trace_hashes_sha256"],
            "schedule_hashes_sha256": train_teacher["schedule_hashes_sha256"],
        },
        "seeds": seed_results,
        "aggregate_validation": {
            "seed_count": len(seed_results),
            "mean_of_seed_mean_ratios": float(np.mean(seed_ratios)),
            "std_of_seed_mean_ratios": float(np.std(seed_ratios)),
            "min_seed_mean_ratio": float(np.min(seed_ratios)),
            "max_seed_mean_ratio": float(np.max(seed_ratios)),
            "improved_seed_count": sum(
                int(result["improved_over_warm_start"])
                for result in seed_results
            ),
            "warm_start_fallback_count": sum(
                int(result["selected_warm_start"]) for result in seed_results
            ),
            "failure_count": sum(
                int(result["best_validation"]["failure_count"])
                for result in seed_results
            ),
            "illegal_action_count": sum(
                int(result["best_validation"]["illegal_action_count"])
                for result in seed_results
            ),
        },
        "selection": {
            **config["selection"],
            "split": "validation",
            "test_accessed": False,
        },
        "resumability": {
            "state_format_version": 1,
            "boundary": "completed_ppo_epoch",
            "state_artifacts": [
                result["training_state"]["name"] for result in seed_results
            ],
            "test_accessed": False,
        },
        "validation_gate_passed": validation_gate_passed,
        "recommendation": (
            "proceed_to_independent_review"
            if validation_gate_passed
            else "retain_masked_mlp_baseline_and_tune_only_on_validation"
        ),
        "run_manifest": "task_gnn_run_manifest.json",
    }
    summary_path = output_dir / "task_gnn_summary.json"
    _write_json(summary_path, summary)
    artifact_names.append(summary_path.name)

    print("[6/6] writing the task-GNN reproducibility manifest")
    artifacts = {
        name: {
            "bytes": (output_dir / name).stat().st_size,
            "sha256": _file_hash(output_dir / name),
        }
        for name in sorted(artifact_names)
    }
    lockfile = repository / "requirements-lock.txt"
    run_manifest = {
        "format_version": 1,
        "mode": "stg_task_gnn_ppo",
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "code": code_metadata,
        "runtime": {
            "trisched": __version__,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "execution": {
            "resume_requested": resume,
            "resumed_seeds": resumed_seeds,
            "resume_boundary": "completed_ppo_epoch",
            "publication_mode": (
                "staging_directory_swap" if resume else "direct_new_directory"
            ),
        },
        "inputs": {
            "config": {
                "name": config_source.name,
                "sha256": _file_hash(config_source),
            },
            "normalized_config_sha256": _canonical_sha256(config),
            "benchmark_manifest": {
                "name": manifest_path.name,
                "sha256": manifest_sha256,
            },
            "dependency_lock": {
                "name": lockfile.name,
                "sha256": _file_hash(lockfile),
            }
            if lockfile.is_file()
            else None,
            "splits": {
                "train": benchmark_manifest["splits"]["train"],
                "validation": benchmark_manifest["splits"]["validation"],
            },
            "architecture": architecture_metadata,
            "feature_names": config["features"]["selected"],
            "seeds": config["seeds"],
            "test_accessed": False,
        },
        "checkpoints": {
            str(result["seed"]): {
                "best": result["best_checkpoint"],
                "last": result["last_checkpoint"],
                "training_state": result["training_state"],
            }
            for result in seed_results
        },
        "artifacts": artifacts,
    }
    _write_json(output_dir / "task_gnn_run_manifest.json", run_manifest)
    print(
        "done: task-GNN seed mean ratios="
        + ", ".join(f"{ratio:.6f}" for ratio in seed_ratios)
        + f", validation_gate_passed={validation_gate_passed}"
    )
    print(f"summary: {summary_path.resolve()}")
    return summary_path


def _task_gnn_resume_transaction_path(output_dir: Path, role: str) -> Path:
    return output_dir.with_name(
        f".{output_dir.name}.task-gnn-resume-{role}-{uuid.uuid4().hex}"
    )


def _publish_task_gnn_resume_staging(
    staging_dir: Path,
    output_dir: Path,
) -> None:
    backup_dir = _task_gnn_resume_transaction_path(output_dir, "backup")
    try:
        os.replace(output_dir, backup_dir)
    except OSError as error:
        try:
            _remove_resume_tree(staging_dir)
        except OSError:
            pass
        _fail(
            "task_gnn_resume_publish",
            "$.output_dir",
            f"could not move the previous evidence directory aside: {error}",
        )
    try:
        os.replace(staging_dir, output_dir)
    except OSError as error:
        rollback_error: OSError | None = None
        try:
            os.replace(backup_dir, output_dir)
        except OSError as captured:
            rollback_error = captured
        try:
            _remove_resume_tree(staging_dir)
        except OSError:
            pass
        _fail(
            "task_gnn_resume_publish",
            "$.output_dir",
            f"could not publish the completed resume transaction: {error}",
            details={
                "rollback_error": str(rollback_error)
                if rollback_error is not None
                else None,
                "backup_dir": str(backup_dir) if rollback_error is not None else None,
            },
        )
    try:
        _remove_resume_tree(backup_dir)
    except OSError as error:
        _fail(
            "task_gnn_resume_publish",
            "$.output_dir",
            f"published resume output but could not remove its backup: {error}",
            details={"backup_dir": str(backup_dir)},
        )


def run_task_gnn_pipeline(
    config_path: str | Path,
    output_override: str | Path | None = None,
    *,
    resume: bool = False,
) -> Path:
    """Run the no-test multi-seed task-GNN pipeline with transactional resume."""

    if not resume:
        return _run_task_gnn_pipeline_in_directory(
            config_path,
            output_override,
            resume=False,
        )

    config_source = Path(config_path).resolve()
    config = load_task_gnn_config(config_source)
    output_dir = (
        _resolve_config_path(config_source, config["output_dir"])
        if output_override is None
        else Path(output_override).resolve()
    )
    if not output_dir.is_dir() or not any(output_dir.iterdir()):
        _fail(
            "task_gnn_resume_output_missing",
            "$.output_dir",
            f"resume output does not exist or is empty: {output_dir}",
        )

    staging_dir = _task_gnn_resume_transaction_path(output_dir, "staging")
    try:
        shutil.copytree(output_dir, staging_dir)
    except (OSError, shutil.Error) as error:
        try:
            _remove_resume_tree(staging_dir)
        except OSError:
            pass
        _fail(
            "task_gnn_resume_stage_write",
            "$.output_dir",
            f"could not create resume transaction staging: {error}",
        )

    try:
        staged_summary = _run_task_gnn_pipeline_in_directory(
            config_source,
            staging_dir,
            resume=True,
        )
    except BaseException:
        try:
            _remove_resume_tree(staging_dir)
        except OSError:
            pass
        raise

    _publish_task_gnn_resume_staging(staging_dir, output_dir)
    return output_dir / staged_summary.name
