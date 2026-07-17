from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .benchmark import load_benchmark_manifest, load_frozen_split
from .env import HeterogeneousDagEnv, ScheduleResult, validate_schedule
from .learning import (
    FEATURE_NAMES,
    DistributionCache,
    MaskedMLPPolicy,
    build_candidate_feature_context,
    candidate_features,
)
from .oracle import independent_heft_schedule, validate_schedule_independent
from .policies import HeftPolicy, compute_upward_ranks
from .scenario import Scenario


class BehaviorCloningError(ValueError):
    """Stable diagnostic for the frozen-teacher behavior-cloning pipeline."""

    def __init__(
        self,
        code: str,
        path: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.path = path
        self.message = message
        self.details = dict(details or {})
        super().__init__(f"{code} at {path}: {message}")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "path": self.path,
            "message": self.message,
        }
        if self.details:
            payload["details"] = self.details
        return payload


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


def _json_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _git_metadata(repository: Path) -> dict[str, Any]:
    environment_head = os.environ.get("TRISCHED_GIT_HEAD", "").strip()
    if environment_head:
        dirty_text = os.environ.get("TRISCHED_GIT_DIRTY", "").strip().lower()
        dirty = {
            "true": True,
            "1": True,
            "false": False,
            "0": False,
        }.get(dirty_text)
        return {
            "commit": environment_head,
            "working_tree_dirty": dirty,
            "source": "TRISCHED_GIT_HEAD",
        }
    try:
        commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {
            "commit": None,
            "working_tree_dirty": None,
            "source": "unavailable",
        }
    return {
        "commit": commit,
        "working_tree_dirty": bool(status.strip()),
        "source": "git",
    }


def policy_parameter_hash(policy: MaskedMLPPolicy) -> str:
    """Hash inference parameters independently from NPZ container metadata."""

    digest = hashlib.sha256()
    schema = {
        "format_version": 1,
        "hidden_dim": policy.hidden_dim,
        "feature_names": list(FEATURE_NAMES),
        "parameters": [
            {
                "name": name,
                "shape": list(policy.params[name].shape),
                "dtype": "float64-little-endian",
            }
            for name in sorted(policy.params)
        ],
    }
    digest.update(
        json.dumps(
            schema,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for name in sorted(policy.params):
        digest.update(name.encode("utf-8"))
        values = np.asarray(policy.params[name], dtype="<f8", order="C")
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _positive_integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail("config_value", path, "expected a positive integer")
    return value


def _finite_number(value: Any, path: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("config_value", path, "expected a number")
    result = float(value)
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        qualifier = "positive and finite" if positive else "non-negative and finite"
        _fail("config_value", path, f"expected a {qualifier} number")
    return result


def load_bc_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail("config_read", "$", str(error))
    if not isinstance(payload, dict):
        _fail("config_type", "$", "expected an object")
    if payload.get("format_version") != 1:
        _fail("config_version", "$.format_version", "expected 1")
    seed = payload.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        _fail("config_value", "$.seed", "expected an integer")
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
    training = payload.get("training")
    if not isinstance(training, dict):
        _fail("config_type", "$.training", "expected an object")
    normalized_training = {
        "hidden_dim": _positive_integer(
            training.get("hidden_dim", 32), "$.training.hidden_dim"
        ),
        "epochs": _positive_integer(
            training.get("epochs", 3), "$.training.epochs"
        ),
        "learning_rate": _finite_number(
            training.get("learning_rate", 0.004),
            "$.training.learning_rate",
            positive=True,
        ),
        "gradient_clip": _finite_number(
            training.get("gradient_clip", 5.0),
            "$.training.gradient_clip",
            positive=False,
        ),
        "shuffle_seed_offset": _positive_integer(
            training.get("shuffle_seed_offset", 101),
            "$.training.shuffle_seed_offset",
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
    return {
        "format_version": 1,
        "seed": seed,
        "output_dir": output_dir,
        "benchmark": {
            "manifest": benchmark["manifest"],
            "raw_root": benchmark["raw_root"],
        },
        "training": normalized_training,
        "selection": {
            "metric": metric,
            "failure_penalty_ratio": failure_penalty,
            "tie_break": [
                "zero_failures",
                "higher_teacher_action_accuracy",
                "earlier_epoch",
            ],
        },
    }


def _schedule_payload(scenario_hash: str, result: Any) -> dict[str, Any]:
    return {
        "scenario_hash": scenario_hash,
        "makespan": float(result.makespan),
        "entries": [
            {
                "task_id": int(entry.task_id),
                "resource_id": int(entry.resource_id),
                "start": float(entry.start),
                "finish": float(entry.finish),
            }
            for entry in result.entries
        ],
    }


def _assert_oracle_match(
    scenario: Scenario,
    production: ScheduleResult,
    independent: Any,
    *,
    tolerance: float = 1e-7,
) -> None:
    if len(production.entries) != len(independent.entries):
        _fail(
            "teacher_oracle_mismatch",
            f"$.scenarios.{scenario.id}",
            "production and independent schedules have different lengths",
        )
    for index, (left, right) in enumerate(
        zip(production.entries, independent.entries)
    ):
        if (
            left.task_id != right.task_id
            or left.resource_id != right.resource_id
            or abs(left.start - right.start) > tolerance
            or abs(left.finish - right.finish) > tolerance
        ):
            _fail(
                "teacher_oracle_mismatch",
                f"$.scenarios.{scenario.id}.entries[{index}]",
                "production and independent HEFT differ",
            )
    if abs(production.makespan - independent.makespan) > tolerance:
        _fail(
            "teacher_oracle_mismatch",
            f"$.scenarios.{scenario.id}.makespan",
            "production and independent HEFT differ",
        )


def build_teacher_manifest(
    scenarios: Sequence[Scenario],
    source_entries: Sequence[Mapping[str, Any]],
    *,
    split: str,
    purpose: str,
    benchmark_manifest_name: str,
    benchmark_manifest_sha256: str,
    benchmark_id: str,
    code_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    allowed = {
        "behavior_cloning_teacher": "train",
        "model_selection_reference": "validation",
    }
    if allowed.get(purpose) != split:
        _fail(
            "teacher_split_usage",
            "$.split",
            f"purpose {purpose!r} cannot use split {split!r}",
        )
    if len(scenarios) != len(source_entries):
        _fail(
            "teacher_source_count",
            "$.entries",
            "scenario and source entry counts differ",
        )
    records: list[dict[str, Any]] = []
    total_actions = 0
    for index, (scenario, source_entry) in enumerate(
        zip(scenarios, source_entries)
    ):
        if source_entry.get("split") != split:
            _fail(
                "teacher_split_usage",
                f"$.entries[{index}].split",
                "source entry belongs to a different split",
            )
        scenario_hash = scenario.content_hash()
        if (
            source_entry.get("scenario_id") != scenario.id
            or source_entry.get("scenario_hash") != scenario_hash
        ):
            _fail(
                "teacher_source_mismatch",
                f"$.entries[{index}]",
                "verified Scenario does not match frozen source entry",
            )
        production = _run_heft(scenario)
        independent = independent_heft_schedule(scenario)
        validate_schedule_independent(scenario, independent)
        _assert_oracle_match(scenario, production, independent)
        schedule_payload = _schedule_payload(scenario_hash, production)
        actions = [
            [entry.task_id, entry.resource_id]
            for entry in production.entries
        ]
        total_actions += len(actions)
        records.append(
            {
                "split_index": int(source_entry["split_index"]),
                "source": str(source_entry["source"]),
                "source_sha256": str(source_entry["source_sha256"]),
                "scenario_id": scenario.id,
                "scenario_hash": scenario_hash,
                "task_count": scenario.task_count,
                "actions": actions,
                "makespan": float(production.makespan),
                "trace_hash": _json_hash(
                    {"scenario_hash": scenario_hash, "actions": actions}
                ),
                "schedule_hash": _json_hash(schedule_payload),
                "production_independent_match": True,
            }
        )
    trace_hashes = [record["trace_hash"] for record in records]
    schedule_hashes = [record["schedule_hash"] for record in records]
    return {
        "format_version": 1,
        "kind": "heft_teacher_traces",
        "purpose": purpose,
        "split": split,
        "test_accessed": False,
        "benchmark": {
            "id": benchmark_id,
            "manifest": benchmark_manifest_name,
            "manifest_sha256": benchmark_manifest_sha256,
        },
        "generator": {
            "production": "trisched.policies.HeftPolicy",
            "independent": "trisched.oracle.independent_heft_schedule",
            "both_validators_required": True,
            "code": dict(code_metadata),
        },
        "scenario_count": len(records),
        "action_count": total_actions,
        "failure_count": 0,
        "trace_hashes_sha256": _json_hash(trace_hashes),
        "schedule_hashes_sha256": _json_hash(schedule_hashes),
        "entries": records,
    }


def _run_heft(scenario: Scenario) -> ScheduleResult:
    env = HeterogeneousDagEnv(scenario)
    policy = HeftPolicy()
    policy.reset(scenario)
    while not env.done:
        action = policy.select_action(env)
        if action not in env.candidate_actions():
            _fail(
                "teacher_illegal_action",
                f"$.scenarios.{scenario.id}",
                f"HEFT selected illegal action {action}",
            )
        env.step(*action)
    result = env.result(policy.name)
    validate_schedule(scenario, result)
    validate_schedule_independent(scenario, result)
    return result


def _record_map(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        _fail("teacher_manifest", "$.entries", "expected an array")
    result: dict[str, Mapping[str, Any]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not isinstance(
            entry.get("scenario_id"), str
        ):
            _fail(
                "teacher_manifest",
                f"$.entries[{index}]",
                "expected a teacher entry",
            )
        scenario_id = entry["scenario_id"]
        if scenario_id in result:
            _fail(
                "teacher_manifest",
                f"$.entries[{index}].scenario_id",
                "duplicate scenario",
            )
        result[scenario_id] = entry
    return result


def _teacher_actions(
    scenario: Scenario,
    record: Mapping[str, Any],
) -> tuple[tuple[int, int], ...]:
    raw_actions = record.get("actions")
    if not isinstance(raw_actions, list):
        _fail(
            "teacher_manifest",
            f"$.entries.{scenario.id}.actions",
            "expected an array",
        )
    actions: list[tuple[int, int]] = []
    for index, raw in enumerate(raw_actions):
        if (
            not isinstance(raw, list)
            or len(raw) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in raw)
        ):
            _fail(
                "teacher_manifest",
                f"$.entries.{scenario.id}.actions[{index}]",
                "expected [task_id, resource_id]",
            )
        actions.append((raw[0], raw[1]))
    if len(actions) != scenario.task_count:
        _fail(
            "teacher_manifest",
            f"$.entries.{scenario.id}.actions",
            "teacher trace must contain one action per task",
        )
    expected_trace = _json_hash(
        {"scenario_hash": scenario.content_hash(), "actions": raw_actions}
    )
    if record.get("trace_hash") != expected_trace:
        _fail(
            "teacher_trace_hash",
            f"$.entries.{scenario.id}.trace_hash",
            "teacher trace changed",
        )
    return tuple(actions)


FrozenTeacherStates = tuple[tuple[np.ndarray, int], ...]


def _freeze_teacher_states(
    scenario: Scenario,
    record: Mapping[str, Any],
) -> FrozenTeacherStates:
    """Materialize policy-independent candidate features once per trace."""

    env = HeterogeneousDagEnv(scenario)
    ranks = compute_upward_ranks(scenario)
    context = build_candidate_feature_context(scenario, ranks)
    states: list[tuple[np.ndarray, int]] = []
    actions = _teacher_actions(scenario, record)
    for step, action in enumerate(actions):
        candidates, features = candidate_features(env, ranks, context)
        if action not in candidates:
            _fail(
                "teacher_illegal_action",
                f"$.entries.{scenario.id}.actions[{step}]",
                "frozen action is not legal while materializing features",
            )
        states.append((features, candidates.index(action)))
        env.step(*action)
    result = env.result("frozen_heft_teacher_features")
    validate_schedule(scenario, result)
    validate_schedule_independent(scenario, result)
    if abs(result.makespan - float(record["makespan"])) > 1e-7:
        _fail(
            "teacher_replay_mismatch",
            f"$.entries.{scenario.id}.makespan",
            "materialized teacher makespan changed",
        )
    return tuple(states)


def _distribution_from_features(
    policy: MaskedMLPPolicy,
    features: np.ndarray,
) -> DistributionCache:
    hidden = np.tanh(features @ policy.params["w1"] + policy.params["b1"])
    scores = hidden @ policy.params["w2"]
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    probabilities = exp_scores / np.sum(exp_scores)
    return DistributionCache((), features, hidden, probabilities, 1.0)


def _train_frozen_episode(
    policy: MaskedMLPPolicy,
    states: FrozenTeacherStates,
    *,
    learning_rate: float,
    gradient_clip: float,
) -> tuple[float, int, int, float]:
    loss_sum = 0.0
    correct = 0
    gradient_norm_sum = 0.0
    for features, target in states:
        cache = _distribution_from_features(policy, features)
        correct += int(np.argmax(cache.probabilities) == target)
        loss_sum -= float(np.log(cache.probabilities[target] + 1e-12))
        gradients = policy.log_probability_gradients(cache, target)
        gradient_norm_sum += policy.apply_gradients(
            gradients,
            learning_rate,
            gradient_clip,
        )
    return loss_sum, correct, len(states), gradient_norm_sum


def _teacher_state_diagnostics(
    policy: MaskedMLPPolicy,
    scenario: Scenario,
    record: Mapping[str, Any],
) -> tuple[float, int, int]:
    env = HeterogeneousDagEnv(scenario)
    policy.reset(scenario)
    loss_sum = 0.0
    correct = 0
    actions = _teacher_actions(scenario, record)
    for step, action in enumerate(actions):
        cache = policy.distribution(env)
        if action not in cache.actions:
            _fail(
                "teacher_illegal_action",
                f"$.entries.{scenario.id}.actions[{step}]",
                "validation reference is not legal during replay",
            )
        target = cache.actions.index(action)
        correct += int(np.argmax(cache.probabilities) == target)
        loss_sum -= float(np.log(cache.probabilities[target] + 1e-12))
        env.step(*action)
    return loss_sum, correct, len(actions)


def _frozen_teacher_state_diagnostics(
    policy: MaskedMLPPolicy,
    states: FrozenTeacherStates,
) -> tuple[float, int, int]:
    loss_sum = 0.0
    correct = 0
    for features, target in states:
        cache = _distribution_from_features(policy, features)
        correct += int(np.argmax(cache.probabilities) == target)
        loss_sum -= float(np.log(cache.probabilities[target] + 1e-12))
    return loss_sum, correct, len(states)


def _policy_rollout(
    policy: MaskedMLPPolicy,
    scenario: Scenario,
) -> tuple[ScheduleResult, int, int]:
    env = HeterogeneousDagEnv(scenario)
    policy.reset(scenario)
    attempts = 0
    illegal = 0
    while not env.done:
        candidates = env.candidate_actions()
        if not candidates:
            raise RuntimeError("no legal action before episode completion")
        action = policy.select_action(env)
        attempts += 1
        if action not in candidates:
            illegal += 1
            raise ValueError(f"policy selected illegal action {action}")
        env.step(*action)
    result = env.result(policy.name)
    validate_schedule(scenario, result)
    validate_schedule_independent(scenario, result)
    return result, attempts, illegal


def evaluate_bc_policy(
    policy: MaskedMLPPolicy,
    scenarios: Sequence[Scenario],
    reference_manifest: Mapping[str, Any],
    *,
    failure_penalty_ratio: float,
    frozen_teacher_states: Mapping[str, FrozenTeacherStates] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if reference_manifest.get("purpose") != "model_selection_reference":
        _fail(
            "validation_reference",
            "$.purpose",
            "validation diagnostics require model_selection_reference",
        )
    if reference_manifest.get("split") != "validation":
        _fail(
            "validation_reference",
            "$.split",
            "validation diagnostics cannot use another split",
        )
    records = _record_map(reference_manifest)
    teacher_loss = 0.0
    teacher_correct = 0
    teacher_actions = 0
    rollout_attempts = 0
    illegal_actions = 0
    ratios: list[float] = []
    success_ratios: list[float] = []
    rows: list[dict[str, Any]] = []
    error_counts: dict[str, int] = {}
    for scenario in scenarios:
        record = records.get(scenario.id)
        if record is None or record.get("scenario_hash") != scenario.content_hash():
            _fail(
                "validation_reference",
                f"$.entries.{scenario.id}",
                "missing or mismatched validation reference",
            )
        if frozen_teacher_states is None:
            loss, correct, count = _teacher_state_diagnostics(
                policy,
                scenario,
                record,
            )
        else:
            states = frozen_teacher_states.get(scenario.id)
            if states is None:
                _fail(
                    "validation_reference",
                    f"$.entries.{scenario.id}",
                    "missing frozen validation teacher states",
                )
            loss, correct, count = _frozen_teacher_state_diagnostics(
                policy,
                states,
            )
        teacher_loss += loss
        teacher_correct += correct
        teacher_actions += count
        try:
            result, attempts, illegal = _policy_rollout(policy, scenario)
            rollout_attempts += attempts
            illegal_actions += illegal
            ratio = result.makespan / float(record["makespan"])
            ratios.append(ratio)
            success_ratios.append(ratio)
            rows.append(
                {
                    "scenario_id": scenario.id,
                    "scenario_hash": scenario.content_hash(),
                    "status": "success",
                    "teacher_makespan": float(record["makespan"]),
                    "policy_makespan": result.makespan,
                    "ratio": ratio,
                    "score_ratio": ratio,
                    "illegal_action_count": illegal,
                    "error": None,
                }
            )
        except Exception as error:
            code = type(error).__name__
            error_counts[code] = error_counts.get(code, 0) + 1
            ratios.append(failure_penalty_ratio)
            rows.append(
                {
                    "scenario_id": scenario.id,
                    "scenario_hash": scenario.content_hash(),
                    "status": "failure",
                    "teacher_makespan": float(record["makespan"]),
                    "policy_makespan": None,
                    "ratio": None,
                    "score_ratio": failure_penalty_ratio,
                    "illegal_action_count": 1
                    if "illegal action" in str(error)
                    else 0,
                    "error": {"type": code, "message": str(error)},
                }
            )
            if "illegal action" in str(error):
                illegal_actions += 1
                rollout_attempts += 1
    failure_count = len(rows) - len(success_ratios)
    metrics = {
        "scenario_count": len(rows),
        "success_count": len(success_ratios),
        "failure_count": failure_count,
        "failure_rate": failure_count / len(rows),
        "valid_schedule_rate": len(success_ratios) / len(rows),
        "mean_ratio": float(np.mean(ratios)),
        "mean_success_ratio": (
            float(np.mean(success_ratios)) if success_ratios else None
        ),
        "teacher_state_action_count": teacher_actions,
        "teacher_action_accuracy": teacher_correct / teacher_actions,
        "teacher_cross_entropy": teacher_loss / teacher_actions,
        "rollout_action_count": rollout_attempts,
        "illegal_action_count": illegal_actions,
        "illegal_action_rate": (
            illegal_actions / rollout_attempts if rollout_attempts else 0.0
        ),
        "error_counts": dict(sorted(error_counts.items())),
    }
    return metrics, rows


def _snapshot(policy: MaskedMLPPolicy) -> dict[str, np.ndarray]:
    return {name: value.copy() for name, value in policy.params.items()}


def _policy_from_snapshot(
    snapshot: Mapping[str, np.ndarray],
    *,
    hidden_dim: int,
    seed: int,
) -> MaskedMLPPolicy:
    policy = MaskedMLPPolicy(hidden_dim=hidden_dim, seed=seed, deterministic=True)
    for name in policy.params:
        policy.params[name] = np.asarray(snapshot[name], dtype=np.float64).copy()
    return policy


def train_bc_baseline(
    train_scenarios: Sequence[Scenario],
    train_teacher_manifest: Mapping[str, Any],
    validation_scenarios: Sequence[Scenario],
    validation_reference_manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[
    MaskedMLPPolicy,
    MaskedMLPPolicy,
    dict[str, Any],
]:
    if train_teacher_manifest.get("purpose") != "behavior_cloning_teacher":
        _fail(
            "training_teacher",
            "$.purpose",
            "training requires behavior_cloning_teacher",
        )
    if train_teacher_manifest.get("split") != "train":
        _fail("training_split", "$.split", "training is restricted to train")
    if validation_reference_manifest.get("split") != "validation":
        _fail(
            "selection_split",
            "$.split",
            "model selection is restricted to validation",
        )
    train_records = _record_map(train_teacher_manifest)
    if set(train_records) != {scenario.id for scenario in train_scenarios}:
        _fail(
            "training_teacher",
            "$.entries",
            "teacher scenarios do not exactly match train scenarios",
        )
    policy = MaskedMLPPolicy(
        hidden_dim=int(config["hidden_dim"]),
        seed=seed,
        deterministic=True,
    )
    rng = np.random.default_rng(seed + int(config["shuffle_seed_offset"]))
    train_states = {
        scenario.id: _freeze_teacher_states(
            scenario,
            train_records[scenario.id],
        )
        for scenario in train_scenarios
    }
    validation_records = _record_map(validation_reference_manifest)
    if set(validation_records) != {
        scenario.id for scenario in validation_scenarios
    }:
        _fail(
            "validation_reference",
            "$.entries",
            "reference scenarios do not exactly match validation scenarios",
        )
    validation_states = {
        scenario.id: _freeze_teacher_states(
            scenario,
            validation_records[scenario.id],
        )
        for scenario in validation_scenarios
    }
    history: list[dict[str, Any]] = []
    best_key: tuple[float, ...] | None = None
    best_epoch = 0
    best_snapshot: dict[str, np.ndarray] | None = None
    for epoch in range(1, int(config["epochs"]) + 1):
        loss_sum = 0.0
        correct = 0
        action_count = 0
        gradient_norm_sum = 0.0
        for raw_index in rng.permutation(len(train_scenarios)):
            scenario = train_scenarios[int(raw_index)]
            loss, episode_correct, episode_actions, gradient_norm = (
                _train_frozen_episode(
                    policy,
                    train_states[scenario.id],
                    learning_rate=float(config["learning_rate"]),
                    gradient_clip=float(config["gradient_clip"]),
                )
            )
            loss_sum += loss
            correct += episode_correct
            action_count += episode_actions
            gradient_norm_sum += gradient_norm
        validation, _ = evaluate_bc_policy(
            policy,
            validation_scenarios,
            validation_reference_manifest,
            failure_penalty_ratio=float(config["failure_penalty_ratio"]),
            frozen_teacher_states=validation_states,
        )
        record = {
            "epoch": epoch,
            "train_action_count": action_count,
            "train_cross_entropy": loss_sum / action_count,
            "train_action_accuracy": correct / action_count,
            "train_illegal_action_count": 0,
            "mean_gradient_norm": gradient_norm_sum / action_count,
            "validation": validation,
        }
        history.append(record)
        key = (
            float(validation["failure_count"]),
            float(validation["mean_ratio"]),
            -float(validation["teacher_action_accuracy"]),
            float(epoch),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_epoch = epoch
            best_snapshot = _snapshot(policy)
    assert best_snapshot is not None
    best_policy = _policy_from_snapshot(
        best_snapshot,
        hidden_dim=policy.hidden_dim,
        seed=seed,
    )
    last_policy = _policy_from_snapshot(
        _snapshot(policy),
        hidden_dim=policy.hidden_dim,
        seed=seed,
    )
    return best_policy, last_policy, {
        "format_version": 1,
        "algorithm": "frozen HEFT behavior cloning",
        "feature_names": list(FEATURE_NAMES),
        "seed": seed,
        "epochs": history,
        "selection": {
            "split": "validation",
            "test_accessed": False,
            "metric": "validation_mean_ratio",
            "tie_break": [
                "zero_failures",
                "higher_teacher_action_accuracy",
                "earlier_epoch",
            ],
            "best_epoch": best_epoch,
        },
    }


def _resolve_config_path(config_source: Path, value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = config_source.parent / candidate
    return candidate.resolve()


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            )


def run_bc_pipeline(
    config_path: str | Path,
    output_override: str | Path | None = None,
) -> Path:
    """Generate frozen train teachers and select a pure BC checkpoint on validation."""

    config_source = Path(config_path).resolve()
    config = load_bc_config(config_source)
    manifest_path = _resolve_config_path(
        config_source, config["benchmark"]["manifest"]
    )
    raw_root = _resolve_config_path(
        config_source, config["benchmark"]["raw_root"]
    )
    if output_override is None:
        output_dir = _resolve_config_path(config_source, config["output_dir"])
    else:
        output_dir = Path(output_override).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        _fail(
            "output_not_empty",
            "$.output_dir",
            f"refusing to mix evidence in {output_dir}",
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "resolved_config.json", config)

    print("[1/5] loading verified train and validation splits (test forbidden)")
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
        entry
        for entry in benchmark_manifest["entries"]
        if entry["split"] == "train"
    ]
    validation_entries = [
        entry
        for entry in benchmark_manifest["entries"]
        if entry["split"] == "validation"
    ]
    repository = Path(__file__).resolve().parents[1]
    code_metadata = _git_metadata(repository)

    print("[2/5] generating HEFT traces with production/oracle agreement")
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
    _write_json(output_dir / "train_teacher_manifest.json", train_teacher)
    _write_json(
        output_dir / "validation_reference_manifest.json",
        validation_reference,
    )
    _write_jsonl(output_dir / "teacher_failures.jsonl", [])

    print("[3/5] training pure behavior cloning and selecting on validation")
    training_config = {
        **config["training"],
        "failure_penalty_ratio": config["selection"][
            "failure_penalty_ratio"
        ],
    }
    best_policy, last_policy, curve = train_bc_baseline(
        train_scenarios,
        train_teacher,
        validation_scenarios,
        validation_reference,
        training_config,
        seed=int(config["seed"]),
    )
    best_path = output_dir / "bc_best.npz"
    last_path = output_dir / "bc_last.npz"
    best_policy.save(best_path)
    last_policy.save(last_path)
    _write_json(output_dir / "bc_training_curve.json", curve)

    print("[4/5] validating best/last checkpoints without test access")
    best_metrics, best_rows = evaluate_bc_policy(
        best_policy,
        validation_scenarios,
        validation_reference,
        failure_penalty_ratio=config["selection"]["failure_penalty_ratio"],
    )
    last_metrics, last_rows = evaluate_bc_policy(
        last_policy,
        validation_scenarios,
        validation_reference,
        failure_penalty_ratio=config["selection"]["failure_penalty_ratio"],
    )
    failures = [row for row in best_rows if row["status"] == "failure"]
    _write_jsonl(output_dir / "validation_failures.jsonl", failures)
    diagnostics = {
        "format_version": 1,
        "split": "validation",
        "test_accessed": False,
        "selection_epoch": curve["selection"]["best_epoch"],
        "best": {"metrics": best_metrics, "per_instance": best_rows},
        "last": {"metrics": last_metrics, "per_instance": last_rows},
    }
    _write_json(
        output_dir / "bc_validation_diagnostics.json",
        diagnostics,
    )
    best_checkpoint = {
        "name": best_path.name,
        "sha256": _file_hash(best_path),
        "parameter_sha256": policy_parameter_hash(best_policy),
    }
    last_checkpoint = {
        "name": last_path.name,
        "sha256": _file_hash(last_path),
        "parameter_sha256": policy_parameter_hash(last_policy),
    }
    summary = {
        "format_version": 1,
        "mode": "stg_behavior_cloning",
        "benchmark_id": benchmark_manifest["benchmark_id"],
        "data_access": {
            "loaded_splits": ["train", "validation"],
            "teacher_split": "train",
            "model_selection_split": "validation",
            "test_accessed": False,
            "test_status": "forbidden_until_final_evaluation",
        },
        "teacher": {
            "scenario_count": train_teacher["scenario_count"],
            "action_count": train_teacher["action_count"],
            "failure_count": train_teacher["failure_count"],
            "trace_hashes_sha256": train_teacher[
                "trace_hashes_sha256"
            ],
            "schedule_hashes_sha256": train_teacher[
                "schedule_hashes_sha256"
            ],
        },
        "selection": curve["selection"],
        "best_checkpoint": best_checkpoint,
        "last_checkpoint": last_checkpoint,
        "best_validation": best_metrics,
        "last_validation": last_metrics,
        "publishable": best_metrics["failure_count"] == 0
        and best_metrics["illegal_action_count"] == 0,
        "run_manifest": "bc_run_manifest.json",
    }
    _write_json(output_dir / "bc_summary.json", summary)

    print("[5/5] writing reproducibility manifest")
    artifact_names = [
        "resolved_config.json",
        "train_teacher_manifest.json",
        "validation_reference_manifest.json",
        "teacher_failures.jsonl",
        "bc_training_curve.json",
        "bc_best.npz",
        "bc_last.npz",
        "validation_failures.jsonl",
        "bc_validation_diagnostics.json",
        "bc_summary.json",
    ]
    artifacts = {
        name: {
            "bytes": (output_dir / name).stat().st_size,
            "sha256": _file_hash(output_dir / name),
        }
        for name in artifact_names
    }
    lockfile = repository / "requirements-lock.txt"
    run_manifest = {
        "format_version": 1,
        "mode": "stg_behavior_cloning",
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
            "test_accessed": False,
        },
        "checkpoints": {
            "best": best_checkpoint,
            "last": last_checkpoint,
        },
        "artifacts": dict(sorted(artifacts.items())),
    }
    _write_json(output_dir / "bc_run_manifest.json", run_manifest)
    print(
        "done: validation mean_ratio="
        f"{best_metrics['mean_ratio']:.6f}, "
        f"illegal_action_rate={best_metrics['illegal_action_rate']:.6f}"
    )
    print(f"summary: {(output_dir / 'bc_summary.json').resolve()}")
    return output_dir / "bc_summary.json"
