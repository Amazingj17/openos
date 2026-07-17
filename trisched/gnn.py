from __future__ import annotations

import copy
import hashlib
import json
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
class FrozenTaskGraph:
    scenario_id: str
    scenario_sha256: str
    task_count: int
    ranks: np.ndarray
    feature_context: CandidateFeatureContext
    node_features: np.ndarray
    predecessor_adjacency: np.ndarray
    successor_adjacency: np.ndarray


@dataclass(frozen=True)
class FrozenTaskGNNState:
    graph: FrozenTaskGraph
    actions: tuple[tuple[int, int], ...]
    features: np.ndarray


@dataclass(frozen=True)
class TaskGNNDistributionCache:
    actions: tuple[tuple[int, int], ...]
    features: np.ndarray
    task_indices: np.ndarray
    node_features: np.ndarray
    predecessor_adjacency: np.ndarray
    successor_adjacency: np.ndarray
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


def _read_only(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def freeze_task_graph(
    scenario: Scenario,
) -> FrozenTaskGraph:
    ranks = compute_upward_ranks(scenario)
    context = build_candidate_feature_context(scenario, ranks)
    predecessor, successor = _mean_adjacency(scenario)
    return FrozenTaskGraph(
        scenario_id=scenario.id,
        scenario_sha256=scenario.content_hash(),
        task_count=scenario.task_count,
        ranks=_read_only(ranks),
        feature_context=context,
        node_features=_read_only(
            _task_node_features(scenario, ranks, context)
        ),
        predecessor_adjacency=_read_only(predecessor),
        successor_adjacency=_read_only(successor),
    )


def freeze_task_gnn_state(
    env: HeterogeneousDagEnv,
    *,
    graph: FrozenTaskGraph | None = None,
) -> FrozenTaskGNNState:
    scenario = env.scenario
    resolved_graph = (
        freeze_task_graph(scenario)
        if graph is None
        else graph
    )
    if (
        resolved_graph.scenario_sha256 != scenario.content_hash()
        or resolved_graph.task_count != scenario.task_count
    ):
        raise ValueError("frozen task graph does not match the environment")
    actions, features = candidate_features(
        env,
        resolved_graph.ranks,
        resolved_graph.feature_context,
    )
    selected = features[
        :,
        [FEATURE_NAMES.index(name) for name in TASK_GNN_FEATURE_NAMES],
    ]
    return FrozenTaskGNNState(
        graph=resolved_graph,
        actions=actions,
        features=_read_only(selected),
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
        self._adam_m = {
            name: np.zeros_like(value) for name, value in self.params.items()
        }
        self._adam_v = {
            name: np.zeros_like(value) for name, value in self.params.items()
        }
        self._adam_step = 0
        self.ranks: np.ndarray | None = None
        self.feature_context: CandidateFeatureContext | None = None
        self._scenario: Scenario | None = None
        self._graph: FrozenTaskGraph | None = None

    @property
    def parameter_count(self) -> int:
        return sum(int(value.size) for value in self.params.values())

    def reset(self, scenario: Scenario) -> None:
        self._scenario = scenario
        self._graph = freeze_task_graph(scenario)
        self.ranks = self._graph.ranks
        self.feature_context = self._graph.feature_context

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
        graph: FrozenTaskGraph,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        node_hidden = np.tanh(
            graph.node_features @ self.params["node_w"]
            + self.params["node_b"]
        )
        predecessor_hidden = graph.predecessor_adjacency @ node_hidden
        successor_hidden = graph.successor_adjacency @ node_hidden
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
        if self._graph is None:
            raise RuntimeError("task-GNN must be reset with a scenario before use")
        return self._distribution_with_graph(
            features,
            actions=actions,
            graph=self._graph,
            temperature=temperature,
        )

    def distribution_from_frozen_state(
        self,
        state: FrozenTaskGNNState,
        *,
        temperature: float = 1.0,
    ) -> TaskGNNDistributionCache:
        return self._distribution_with_graph(
            state.features,
            actions=state.actions,
            graph=state.graph,
            temperature=temperature,
        )

    def _distribution_with_graph(
        self,
        features: np.ndarray,
        *,
        actions: tuple[tuple[int, int], ...],
        graph: FrozenTaskGraph,
        temperature: float,
    ) -> TaskGNNDistributionCache:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        selected = self.select_feature_columns(features)
        if selected.shape[0] == 0:
            raise ValueError("candidate feature matrix must not be empty")
        if len(actions) != selected.shape[0]:
            raise ValueError("candidate actions and features must have equal length")
        task_indices = np.asarray(
            [task_id for task_id, _ in actions], dtype=np.int64
        )
        if np.any(task_indices < 0) or np.any(
            task_indices >= graph.task_count
        ):
            raise ValueError("candidate action contains an unknown task id")
        (
            node_hidden,
            predecessor_hidden,
            successor_hidden,
            node_context,
        ) = self._encode_tasks(graph)
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
            node_features=graph.node_features,
            predecessor_adjacency=graph.predecessor_adjacency,
            successor_adjacency=graph.successor_adjacency,
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

    def empty_gradients(self) -> dict[str, np.ndarray]:
        return {
            name: np.zeros_like(value) for name, value in self.params.items()
        }

    def log_probability_gradients(
        self,
        cache: TaskGNNDistributionCache,
        selected_index: int,
    ) -> dict[str, np.ndarray]:
        if not 0 <= selected_index < len(cache.probabilities):
            raise IndexError("selected_index is outside the masked distribution")
        d_scores = -cache.probabilities.copy()
        d_scores[selected_index] += 1.0
        return self.score_gradients(cache, d_scores)

    def entropy_gradients(
        self,
        cache: TaskGNNDistributionCache,
    ) -> dict[str, np.ndarray]:
        probabilities = cache.probabilities
        log_probabilities = np.log(probabilities + 1e-12)
        entropy = -float(np.sum(probabilities * log_probabilities))
        d_scores = -probabilities * (log_probabilities + entropy)
        return self.score_gradients(cache, d_scores)

    def score_gradients(
        self,
        cache: TaskGNNDistributionCache,
        d_scores: np.ndarray,
    ) -> dict[str, np.ndarray]:
        score_gradient = np.asarray(d_scores, dtype=np.float64).copy()
        if score_gradient.shape != cache.probabilities.shape:
            raise ValueError("score gradient shape does not match probabilities")
        score_gradient /= cache.temperature

        output_w = cache.hidden.T @ score_gradient
        pair_hidden_gradient = (
            score_gradient[:, None] * self.params["output_w"][None, :]
        ) * (1.0 - cache.hidden**2)
        pair_w = cache.features.T @ pair_hidden_gradient
        context_w = (
            cache.node_context[cache.task_indices].T @ pair_hidden_gradient
        )
        pair_b = np.sum(pair_hidden_gradient, axis=0)

        node_context_gradient = np.zeros_like(cache.node_context)
        np.add.at(
            node_context_gradient,
            cache.task_indices,
            pair_hidden_gradient @ self.params["context_w"].T,
        )
        message_gradient = node_context_gradient * (
            1.0 - cache.node_context**2
        )
        message_self_w = cache.node_hidden.T @ message_gradient
        message_predecessor_w = (
            cache.predecessor_hidden.T @ message_gradient
        )
        message_successor_w = cache.successor_hidden.T @ message_gradient
        message_b = np.sum(message_gradient, axis=0)

        node_hidden_gradient = (
            message_gradient @ self.params["message_self_w"].T
            + cache.predecessor_adjacency.T
            @ (message_gradient @ self.params["message_predecessor_w"].T)
            + cache.successor_adjacency.T
            @ (message_gradient @ self.params["message_successor_w"].T)
        )
        node_input_gradient = node_hidden_gradient * (
            1.0 - cache.node_hidden**2
        )
        node_w = cache.node_features.T @ node_input_gradient
        node_b = np.sum(node_input_gradient, axis=0)
        return {
            "node_w": node_w,
            "node_b": node_b,
            "message_self_w": message_self_w,
            "message_predecessor_w": message_predecessor_w,
            "message_successor_w": message_successor_w,
            "message_b": message_b,
            "pair_w": pair_w,
            "context_w": context_w,
            "pair_b": pair_b,
            "output_w": output_w,
        }

    @staticmethod
    def add_gradients(
        target: dict[str, np.ndarray],
        source: dict[str, np.ndarray],
        scale: float = 1.0,
    ) -> None:
        if set(target) != set(source):
            raise ValueError("gradient dictionaries have different parameters")
        for name in target:
            if target[name].shape != source[name].shape:
                raise ValueError(f"gradient shape mismatch for {name!r}")
            target[name] += source[name] * scale

    def apply_gradients(
        self,
        gradients: dict[str, np.ndarray],
        learning_rate: float,
        clip_norm: float,
    ) -> float:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if set(gradients) != set(self.params):
            raise ValueError("gradients do not match task-GNN parameters")
        checked: dict[str, np.ndarray] = {}
        for name, parameter in self.params.items():
            gradient = np.asarray(gradients[name], dtype=np.float64)
            if gradient.shape != parameter.shape or not np.all(
                np.isfinite(gradient)
            ):
                raise ValueError(f"invalid gradient for {name!r}")
            checked[name] = gradient
        norm = float(
            np.sqrt(
                sum(float(np.sum(value * value)) for value in checked.values())
            )
        )
        if norm > clip_norm > 0:
            scale = clip_norm / (norm + 1e-12)
            checked = {name: value * scale for name, value in checked.items()}
        self._adam_step += 1
        beta1, beta2 = 0.9, 0.999
        for name, gradient in checked.items():
            self._adam_m[name] = (
                beta1 * self._adam_m[name] + (1.0 - beta1) * gradient
            )
            self._adam_v[name] = beta2 * self._adam_v[name] + (
                1.0 - beta2
            ) * (gradient * gradient)
            m_hat = self._adam_m[name] / (1.0 - beta1**self._adam_step)
            v_hat = self._adam_v[name] / (1.0 - beta2**self._adam_step)
            self.params[name] += learning_rate * m_hat / (
                np.sqrt(v_hat) + 1e-8
            )
        return norm

    def clone(
        self,
        *,
        deterministic: bool = True,
        include_optimizer: bool = True,
    ) -> TaskGNNPolicy:
        clone = TaskGNNPolicy(
            hidden_dim=self.hidden_dim,
            message_dim=self.message_dim,
            seed=self.seed,
            deterministic=deterministic,
            feature_names=self.feature_names,
        )
        for name in clone.params:
            clone.params[name] = self.params[name].copy()
            if include_optimizer:
                clone._adam_m[name] = self._adam_m[name].copy()
                clone._adam_v[name] = self._adam_v[name].copy()
        if include_optimizer:
            clone._adam_step = self._adam_step
            clone.rng.bit_generator.state = copy.deepcopy(
                self.rng.bit_generator.state
            )
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


def task_gnn_parameter_hash(policy: TaskGNNPolicy) -> str:
    digest = hashlib.sha256()
    schema = {
        "format_version": 1,
        "architecture": policy.architecture,
        "feature_names": list(policy.feature_names),
        "node_feature_names": list(TASK_NODE_FEATURE_NAMES),
        "hidden_dim": policy.hidden_dim,
        "message_dim": policy.message_dim,
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
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for name in sorted(policy.params):
        digest.update(name.encode("utf-8"))
        values = np.asarray(policy.params[name], dtype="<f8", order="C")
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()
