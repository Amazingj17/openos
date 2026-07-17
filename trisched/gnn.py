from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .env import HeterogeneousDagEnv
from .learning import (
    FEATURE_NAMES,
    TEACHER_FEATURE_NAMES,
    CandidateFeatureContext,
    build_candidate_feature_context,
    candidate_features,
)
from .policies import compute_upward_ranks
from .scenario import Scenario


TASK_GNN_FEATURE_NAMES = tuple(
    name for name in FEATURE_NAMES if name not in TEACHER_FEATURE_NAMES
)
TASK_NODE_FEATURE_NAMES = (
    "task_workload",
    "upward_rank",
    "indegree",
    "outdegree",
)


@dataclass(frozen=True)
class TaskGNNDistributionCache:
    actions: tuple[tuple[int, int], ...]
    features: np.ndarray
    task_indices: np.ndarray
    node_features: np.ndarray
    node_hidden: np.ndarray
    predecessor_hidden: np.ndarray
    successor_hidden: np.ndarray
    node_context: np.ndarray
    hidden: np.ndarray
    probabilities: np.ndarray
    temperature: float


def _mean_adjacency(scenario: Scenario) -> tuple[np.ndarray, np.ndarray]:
    """Return row-normalized predecessor and successor aggregation matrices."""

    predecessor = np.zeros(
        (scenario.task_count, scenario.task_count), dtype=np.float64
    )
    successor = np.zeros_like(predecessor)
    for edge in scenario.edges:
        predecessor[edge.target, edge.source] = 1.0
        successor[edge.source, edge.target] = 1.0
    predecessor_degree = np.sum(predecessor, axis=1, keepdims=True)
    successor_degree = np.sum(successor, axis=1, keepdims=True)
    np.divide(
        predecessor,
        predecessor_degree,
        out=predecessor,
        where=predecessor_degree > 0,
    )
    np.divide(
        successor,
        successor_degree,
        out=successor,
        where=successor_degree > 0,
    )
    return predecessor, successor


def _task_node_features(
    scenario: Scenario,
    ranks: np.ndarray,
    context: CandidateFeatureContext,
) -> np.ndarray:
    """Build task nodes from four fields already present in the 14-D input."""

    predecessors = scenario.predecessors()
    successors = scenario.successors()
    return np.asarray(
        [
            [
                task.workload / context.max_workload,
                ranks[task.id] / context.max_rank,
                len(predecessors[task.id]) / context.max_degree,
                len(successors[task.id]) / context.max_degree,
            ]
            for task in scenario.tasks
        ],
        dtype=np.float64,
    )


class TaskGNNPolicy:
    """A masked task-resource scorer with one DAG message-passing layer."""

    name = "task_gnn"
    architecture = "task_gnn_v1"

    def __init__(
        self,
        hidden_dim: int = 32,
        message_dim: int = 8,
        seed: int = 0,
        deterministic: bool = True,
        feature_names: tuple[str, ...] = TASK_GNN_FEATURE_NAMES,
    ) -> None:
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if message_dim <= 0:
            raise ValueError("message_dim must be positive")
        selected_features = tuple(feature_names)
        if selected_features != TASK_GNN_FEATURE_NAMES:
            raise ValueError(
                "task-GNN requires the canonical 14-D teacher-free feature schema"
            )
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim
        self.seed = seed
        self.deterministic = deterministic
        self.feature_names = selected_features
        self.feature_indices = np.asarray(
            [FEATURE_NAMES.index(name) for name in self.feature_names],
            dtype=np.int64,
        )
        self.rng = np.random.default_rng(seed)
        pair_input_dim = len(self.feature_names)
        node_input_dim = len(TASK_NODE_FEATURE_NAMES)
        node_limit = np.sqrt(6.0 / (node_input_dim + message_dim))
        message_limit = np.sqrt(6.0 / (2 * message_dim))
        pair_limit = np.sqrt(6.0 / (pair_input_dim + hidden_dim))
        context_limit = np.sqrt(6.0 / (message_dim + hidden_dim))
        output_limit = np.sqrt(6.0 / (hidden_dim + 1))
        self.params: dict[str, np.ndarray] = {
            "node_w": self.rng.uniform(
                -node_limit,
                node_limit,
                (node_input_dim, message_dim),
            ),
            "node_b": np.zeros(message_dim, dtype=np.float64),
            "message_self_w": self.rng.uniform(
                -message_limit,
                message_limit,
                (message_dim, message_dim),
            ),
            "message_predecessor_w": self.rng.uniform(
                -message_limit,
                message_limit,
                (message_dim, message_dim),
            ),
            "message_successor_w": self.rng.uniform(
                -message_limit,
                message_limit,
                (message_dim, message_dim),
            ),
            "message_b": np.zeros(message_dim, dtype=np.float64),
            "pair_w": self.rng.uniform(
                -pair_limit,
                pair_limit,
                (pair_input_dim, hidden_dim),
            ),
            "context_w": self.rng.uniform(
                -context_limit,
                context_limit,
                (message_dim, hidden_dim),
            ),
            "pair_b": np.zeros(hidden_dim, dtype=np.float64),
            "output_w": self.rng.uniform(
                -output_limit,
                output_limit,
                hidden_dim,
            ),
        }
        self.ranks: np.ndarray | None = None
        self.feature_context: CandidateFeatureContext | None = None
        self._scenario: Scenario | None = None
        self._node_features: np.ndarray | None = None
        self._predecessor_adjacency: np.ndarray | None = None
        self._successor_adjacency: np.ndarray | None = None

    @property
    def parameter_count(self) -> int:
        return sum(int(value.size) for value in self.params.values())

    def reset(self, scenario: Scenario) -> None:
        self._scenario = scenario
        self.ranks = compute_upward_ranks(scenario)
        self.feature_context = build_candidate_feature_context(
            scenario,
            self.ranks,
        )
        self._node_features = _task_node_features(
            scenario,
            self.ranks,
            self.feature_context,
        )
        (
            self._predecessor_adjacency,
            self._successor_adjacency,
        ) = _mean_adjacency(scenario)

    def select_feature_columns(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("candidate features must be a two-dimensional array")
        if values.shape[1] == len(self.feature_names):
            return values
        if values.shape[1] == len(FEATURE_NAMES):
            return values[:, self.feature_indices]
        raise ValueError(
            "candidate feature width does not match the full or task-GNN schema"
        )

    def _encode_tasks(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if (
            self._node_features is None
            or self._predecessor_adjacency is None
            or self._successor_adjacency is None
        ):
            raise RuntimeError("task-GNN must be reset with a scenario before use")
        node_hidden = np.tanh(
            self._node_features @ self.params["node_w"]
            + self.params["node_b"]
        )
        predecessor_hidden = self._predecessor_adjacency @ node_hidden
        successor_hidden = self._successor_adjacency @ node_hidden
        node_context = np.tanh(
            node_hidden @ self.params["message_self_w"]
            + predecessor_hidden @ self.params["message_predecessor_w"]
            + successor_hidden @ self.params["message_successor_w"]
            + self.params["message_b"]
        )
        return (
            node_hidden,
            predecessor_hidden,
            successor_hidden,
            node_context,
        )

    def distribution_from_features(
        self,
        features: np.ndarray,
        *,
        actions: tuple[tuple[int, int], ...],
        temperature: float = 1.0,
    ) -> TaskGNNDistributionCache:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        selected = self.select_feature_columns(features)
        if selected.shape[0] == 0:
            raise ValueError("candidate feature matrix must not be empty")
        if len(actions) != selected.shape[0]:
            raise ValueError("candidate actions and features must have equal length")
        if self._scenario is None or self._node_features is None:
            raise RuntimeError("task-GNN must be reset with a scenario before use")
        task_indices = np.asarray(
            [task_id for task_id, _ in actions], dtype=np.int64
        )
        if np.any(task_indices < 0) or np.any(
            task_indices >= self._scenario.task_count
        ):
            raise ValueError("candidate action contains an unknown task id")
        (
            node_hidden,
            predecessor_hidden,
            successor_hidden,
            node_context,
        ) = self._encode_tasks()
        hidden = np.tanh(
            selected @ self.params["pair_w"]
            + node_context[task_indices] @ self.params["context_w"]
            + self.params["pair_b"]
        )
        scores = (hidden @ self.params["output_w"]) / temperature
        scores -= np.max(scores)
        exp_scores = np.exp(scores)
        probabilities = exp_scores / np.sum(exp_scores)
        return TaskGNNDistributionCache(
            actions=actions,
            features=selected,
            task_indices=task_indices,
            node_features=self._node_features,
            node_hidden=node_hidden,
            predecessor_hidden=predecessor_hidden,
            successor_hidden=successor_hidden,
            node_context=node_context,
            hidden=hidden,
            probabilities=probabilities,
            temperature=temperature,
        )

    def distribution(
        self,
        env: HeterogeneousDagEnv,
        temperature: float = 1.0,
    ) -> TaskGNNDistributionCache:
        if self._scenario is not env.scenario:
            self.reset(env.scenario)
        assert self.ranks is not None
        actions, features = candidate_features(
            env,
            self.ranks,
            self.feature_context,
        )
        return self.distribution_from_features(
            features,
            actions=actions,
            temperature=temperature,
        )

    def select_action(self, env: HeterogeneousDagEnv) -> tuple[int, int]:
        cache = self.distribution(env)
        if self.deterministic:
            index = int(np.argmax(cache.probabilities))
        else:
            index = int(
                self.rng.choice(len(cache.actions), p=cache.probabilities)
            )
        return cache.actions[index]

    def sample(
        self,
        env: HeterogeneousDagEnv,
        temperature: float = 1.0,
    ) -> tuple[tuple[int, int], int, TaskGNNDistributionCache]:
        cache = self.distribution(env, temperature=temperature)
        index = int(self.rng.choice(len(cache.actions), p=cache.probabilities))
        return cache.actions[index], index, cache

    def clone(self, *, deterministic: bool = True) -> TaskGNNPolicy:
        clone = TaskGNNPolicy(
            hidden_dim=self.hidden_dim,
            message_dim=self.message_dim,
            seed=self.seed,
            deterministic=deterministic,
            feature_names=self.feature_names,
        )
        for name in clone.params:
            clone.params[name] = self.params[name].copy()
        return clone

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            architecture=np.asarray(self.architecture),
            hidden_dim=np.asarray([self.hidden_dim], dtype=np.int64),
            message_dim=np.asarray([self.message_dim], dtype=np.int64),
            seed=np.asarray([self.seed], dtype=np.int64),
            feature_names=np.asarray(self.feature_names),
            node_feature_names=np.asarray(TASK_NODE_FEATURE_NAMES),
            **self.params,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        deterministic: bool = True,
    ) -> TaskGNNPolicy:
        with np.load(path, allow_pickle=False) as data:
            architecture = str(data["architecture"].item())
            if architecture != cls.architecture:
                raise ValueError(f"unsupported task-GNN architecture: {architecture}")
            stored_features = tuple(
                str(item) for item in data["feature_names"].tolist()
            )
            stored_node_features = tuple(
                str(item) for item in data["node_feature_names"].tolist()
            )
            if stored_node_features != TASK_NODE_FEATURE_NAMES:
                raise ValueError("task-GNN node feature schema changed")
            policy = cls(
                hidden_dim=int(data["hidden_dim"][0]),
                message_dim=int(data["message_dim"][0]),
                seed=int(data["seed"][0]),
                deterministic=deterministic,
                feature_names=stored_features,
            )
            for name, expected in policy.params.items():
                values = np.asarray(data[name], dtype=np.float64)
                if values.shape != expected.shape or not np.all(np.isfinite(values)):
                    raise ValueError(
                        f"task-GNN parameter {name!r} has an invalid value"
                    )
                policy.params[name] = values.copy()
        return policy


def task_gnn_metadata(policy: TaskGNNPolicy) -> dict[str, Any]:
    return {
        "architecture": policy.architecture,
        "base_feature_count": len(policy.feature_names),
        "base_feature_names": list(policy.feature_names),
        "node_feature_names": list(TASK_NODE_FEATURE_NAMES),
        "message_passing_steps": 1,
        "message_directions": ["predecessor", "successor"],
        "hidden_dim": policy.hidden_dim,
        "message_dim": policy.message_dim,
        "parameter_count": policy.parameter_count,
    }
