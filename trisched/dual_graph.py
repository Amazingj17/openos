from __future__ import annotations

import copy
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .env import HeterogeneousDagEnv, run_policy
from .learning import (
    FEATURE_NAMES,
    TEACHER_FEATURE_NAMES,
    CandidateFeatureContext,
    build_candidate_feature_context,
    candidate_features,
)
from .policies import (
    CpopPolicy,
    GreedyEarliestFinishPolicy,
    HeftPolicy,
    compute_upward_ranks,
)
from .scenario import Scenario, generate_complex_scenario


BASE_PAIR_FEATURE_NAMES = tuple(
    name for name in FEATURE_NAMES if name not in TEACHER_FEATURE_NAMES
)
DUAL_GRAPH_PAIR_FEATURE_NAMES = BASE_PAIR_FEATURE_NAMES + (
    "cpu_core_demand_ratio",
    "memory_demand_ratio",
    "task_accelerator_required",
    "resource_has_accelerator",
    "required_feature_ratio",
)
TASK_NODE_FEATURE_NAMES = (
    "workload",
    "upward_rank",
    "indegree",
    "outdegree",
    "memory_required",
    "cpu_cores_required",
    "accelerator_required",
    "is_cpu_task",
    "is_data_task",
    "is_gpu_task",
    "is_generic_task",
)
RESOURCE_NODE_FEATURE_NAMES = (
    "speed",
    "memory_capacity",
    "cpu_cores",
    "has_accelerator",
    "is_device",
    "is_edge",
    "is_cloud",
    "ready_time",
    "assigned_task_ratio",
)


def _read_only(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    result = np.asarray(matrix, dtype=np.float64).copy()
    totals = np.sum(result, axis=1, keepdims=True)
    np.divide(result, totals, out=result, where=totals > 0)
    return result


@dataclass(frozen=True)
class FrozenDualGraph:
    scenario_identity: int
    scenario_id: str
    scenario_sha256: str
    ranks: np.ndarray
    feature_context: CandidateFeatureContext
    task_features: np.ndarray
    predecessor_adjacency: np.ndarray
    successor_adjacency: np.ndarray
    resource_static_features: np.ndarray
    resource_adjacency: np.ndarray


@dataclass(frozen=True)
class FrozenDualGraphState:
    graph: FrozenDualGraph
    actions: tuple[tuple[int, int], ...]
    pair_features: np.ndarray
    resource_features: np.ndarray


@dataclass(frozen=True)
class DualGraphDistributionCache:
    actions: tuple[tuple[int, int], ...]
    pair_features: np.ndarray
    task_indices: np.ndarray
    resource_indices: np.ndarray
    task_features: np.ndarray
    predecessor_adjacency: np.ndarray
    successor_adjacency: np.ndarray
    task_hidden: np.ndarray
    predecessor_hidden: np.ndarray
    successor_hidden: np.ndarray
    task_context: np.ndarray
    resource_features: np.ndarray
    resource_adjacency: np.ndarray
    resource_hidden: np.ndarray
    resource_neighbor_hidden: np.ndarray
    resource_context: np.ndarray
    hidden: np.ndarray
    probabilities: np.ndarray
    temperature: float


def freeze_dual_graph(scenario: Scenario) -> FrozenDualGraph:
    """Encode immutable task/resource topology for one scenario.

    Task edges retain both precedence direction and normalized data volume.
    Resource edges retain the relative communication quality derived from
    bandwidth and latency.  The resource nodes remain strictly device, edge,
    or cloud nodes; no HPC-only resource category is introduced.
    """

    ranks = compute_upward_ranks(scenario)
    context = build_candidate_feature_context(scenario, ranks)
    predecessors = scenario.predecessors()
    successors = scenario.successors()
    max_memory = max(resource.memory_capacity for resource in scenario.resources)
    max_required_memory = max(
        max(task.memory_required for task in scenario.tasks), 1.0
    )
    max_required_cores = max(task.cpu_cores_required for task in scenario.tasks)
    max_resource_cores = max(resource.cpu_cores for resource in scenario.resources)
    task_features = np.asarray(
        [
            [
                task.workload / context.max_workload,
                ranks[task.id] / context.max_rank,
                len(predecessors[task.id]) / context.max_degree,
                len(successors[task.id]) / context.max_degree,
                task.memory_required / max_required_memory,
                task.cpu_cores_required / max_required_cores,
                float(task.accelerator_required),
                float(task.task_type == "cpu"),
                float(task.task_type == "data"),
                float(task.task_type == "gpu"),
                float(task.task_type == "generic"),
            ]
            for task in scenario.tasks
        ],
        dtype=np.float64,
    )
    max_edge_data = max((edge.data for edge in scenario.edges), default=0.0)
    predecessor = np.zeros(
        (scenario.task_count, scenario.task_count), dtype=np.float64
    )
    successor = np.zeros_like(predecessor)
    for edge in scenario.edges:
        weight = 1.0 + edge.data / max(max_edge_data, 1.0)
        predecessor[edge.target, edge.source] = weight
        successor[edge.source, edge.target] = weight

    resource_static = np.asarray(
        [
            [
                resource.speed / context.max_speed,
                resource.memory_capacity / max_memory,
                resource.cpu_cores / max_resource_cores,
                float(resource.has_accelerator),
                float(resource.kind == "device"),
                float(resource.kind == "edge"),
                float(resource.kind == "cloud"),
            ]
            for resource in scenario.resources
        ],
        dtype=np.float64,
    )
    bandwidth = np.asarray(scenario.bandwidth, dtype=np.float64)
    latency = np.asarray(scenario.latency, dtype=np.float64)
    max_bandwidth = max(float(np.max(bandwidth)), 1.0)
    max_latency = max(float(np.max(latency)), 1.0)
    resource_adjacency = (bandwidth / max_bandwidth) / (
        1.0 + latency / max_latency
    )
    np.fill_diagonal(resource_adjacency, 0.0)
    return FrozenDualGraph(
        scenario_identity=id(scenario),
        scenario_id=scenario.id,
        scenario_sha256=scenario.content_hash(),
        ranks=_read_only(ranks),
        feature_context=context,
        task_features=_read_only(task_features),
        predecessor_adjacency=_read_only(_normalize_rows(predecessor)),
        successor_adjacency=_read_only(_normalize_rows(successor)),
        resource_static_features=_read_only(resource_static),
        resource_adjacency=_read_only(_normalize_rows(resource_adjacency)),
    )


def freeze_dual_graph_state(
    env: HeterogeneousDagEnv,
    *,
    graph: FrozenDualGraph | None = None,
) -> FrozenDualGraphState:
    scenario = env.scenario
    resolved = freeze_dual_graph(scenario) if graph is None else graph
    if (
        resolved.scenario_identity != id(scenario)
        or resolved.task_features.shape[0] != scenario.task_count
        or resolved.resource_static_features.shape[0] != scenario.resource_count
    ):
        raise ValueError("frozen dual graph does not match the environment")
    actions, full_features = candidate_features(
        env, resolved.ranks, resolved.feature_context
    )
    indices = [FEATURE_NAMES.index(name) for name in BASE_PAIR_FEATURE_NAMES]
    base = full_features[:, indices]
    extra = np.asarray(
        [
            [
                scenario.tasks[task_id].cpu_cores_required
                / scenario.resources[resource_id].cpu_cores,
                scenario.tasks[task_id].memory_required
                / scenario.resources[resource_id].memory_capacity,
                float(scenario.tasks[task_id].accelerator_required),
                float(scenario.resources[resource_id].has_accelerator),
                len(scenario.tasks[task_id].required_features)
                / max(len(scenario.resources[resource_id].features), 1),
            ]
            for task_id, resource_id in actions
        ],
        dtype=np.float64,
    )
    assigned = np.zeros(scenario.resource_count, dtype=np.float64)
    for entry in env.entries.values():
        assigned[entry.resource_id] += 1.0
    dynamic = np.asarray(
        [
            [
                env.resource_ready_time(resource.id)
                / resolved.feature_context.time_scale,
                assigned[resource.id] / scenario.task_count,
            ]
            for resource in scenario.resources
        ],
        dtype=np.float64,
    )
    return FrozenDualGraphState(
        graph=resolved,
        actions=actions,
        pair_features=_read_only(np.concatenate((base, extra), axis=1)),
        resource_features=_read_only(
            np.concatenate((resolved.resource_static_features, dynamic), axis=1)
        ),
    )


class CloudEdgeDualGraphPolicy:
    """Masked Actor that jointly embeds the task DAG and resource graph."""

    name = "dual_graph_gnn"
    display_name = "TriSched-DualGraph-PPO"
    architecture = "cloud_edge_dual_graph_v1"

    def __init__(
        self,
        hidden_dim: int = 32,
        task_message_dim: int = 8,
        resource_message_dim: int = 8,
        seed: int = 0,
        deterministic: bool = True,
    ) -> None:
        if min(hidden_dim, task_message_dim, resource_message_dim) <= 0:
            raise ValueError("network dimensions must be positive")
        self.hidden_dim = hidden_dim
        self.task_message_dim = task_message_dim
        self.resource_message_dim = resource_message_dim
        self.seed = seed
        self.deterministic = deterministic
        self.feature_names = DUAL_GRAPH_PAIR_FEATURE_NAMES
        self.rng = np.random.default_rng(seed)

        def weights(rows: int, columns: int) -> np.ndarray:
            limit = np.sqrt(6.0 / (rows + columns))
            return self.rng.uniform(-limit, limit, (rows, columns))

        self.params: dict[str, np.ndarray] = {
            "task_node_w": weights(len(TASK_NODE_FEATURE_NAMES), task_message_dim),
            "task_node_b": np.zeros(task_message_dim),
            "task_self_w": weights(task_message_dim, task_message_dim),
            "task_predecessor_w": weights(task_message_dim, task_message_dim),
            "task_successor_w": weights(task_message_dim, task_message_dim),
            "task_message_b": np.zeros(task_message_dim),
            "resource_node_w": weights(
                len(RESOURCE_NODE_FEATURE_NAMES), resource_message_dim
            ),
            "resource_node_b": np.zeros(resource_message_dim),
            "resource_self_w": weights(resource_message_dim, resource_message_dim),
            "resource_neighbor_w": weights(
                resource_message_dim, resource_message_dim
            ),
            "resource_message_b": np.zeros(resource_message_dim),
            "pair_w": weights(len(DUAL_GRAPH_PAIR_FEATURE_NAMES), hidden_dim),
            "task_context_w": weights(task_message_dim, hidden_dim),
            "resource_context_w": weights(resource_message_dim, hidden_dim),
            "pair_b": np.zeros(hidden_dim),
            "output_w": weights(hidden_dim, 1).reshape(hidden_dim),
        }
        self._adam_m = {name: np.zeros_like(value) for name, value in self.params.items()}
        self._adam_v = {name: np.zeros_like(value) for name, value in self.params.items()}
        self._adam_step = 0
        self._scenario: Scenario | None = None
        self._graph: FrozenDualGraph | None = None

    @property
    def parameter_count(self) -> int:
        return sum(int(value.size) for value in self.params.values())

    def reset(self, scenario: Scenario) -> None:
        self._scenario = scenario
        self._graph = freeze_dual_graph(scenario)

    def _distribution(
        self, state: FrozenDualGraphState, temperature: float
    ) -> DualGraphDistributionCache:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not state.actions:
            raise ValueError("masked action set must not be empty")
        graph = state.graph
        task_hidden = np.tanh(
            graph.task_features @ self.params["task_node_w"]
            + self.params["task_node_b"]
        )
        predecessor_hidden = graph.predecessor_adjacency @ task_hidden
        successor_hidden = graph.successor_adjacency @ task_hidden
        task_context = np.tanh(
            task_hidden @ self.params["task_self_w"]
            + predecessor_hidden @ self.params["task_predecessor_w"]
            + successor_hidden @ self.params["task_successor_w"]
            + self.params["task_message_b"]
        )
        resource_hidden = np.tanh(
            state.resource_features @ self.params["resource_node_w"]
            + self.params["resource_node_b"]
        )
        resource_neighbor_hidden = graph.resource_adjacency @ resource_hidden
        resource_context = np.tanh(
            resource_hidden @ self.params["resource_self_w"]
            + resource_neighbor_hidden @ self.params["resource_neighbor_w"]
            + self.params["resource_message_b"]
        )
        task_indices = np.asarray([item[0] for item in state.actions], dtype=np.int64)
        resource_indices = np.asarray(
            [item[1] for item in state.actions], dtype=np.int64
        )
        hidden = np.tanh(
            state.pair_features @ self.params["pair_w"]
            + task_context[task_indices] @ self.params["task_context_w"]
            + resource_context[resource_indices]
            @ self.params["resource_context_w"]
            + self.params["pair_b"]
        )
        scores = hidden @ self.params["output_w"] / temperature
        scores -= np.max(scores)
        probabilities = np.exp(scores)
        probabilities /= np.sum(probabilities)
        return DualGraphDistributionCache(
            actions=state.actions,
            pair_features=state.pair_features,
            task_indices=task_indices,
            resource_indices=resource_indices,
            task_features=graph.task_features,
            predecessor_adjacency=graph.predecessor_adjacency,
            successor_adjacency=graph.successor_adjacency,
            task_hidden=task_hidden,
            predecessor_hidden=predecessor_hidden,
            successor_hidden=successor_hidden,
            task_context=task_context,
            resource_features=state.resource_features,
            resource_adjacency=graph.resource_adjacency,
            resource_hidden=resource_hidden,
            resource_neighbor_hidden=resource_neighbor_hidden,
            resource_context=resource_context,
            hidden=hidden,
            probabilities=probabilities,
            temperature=temperature,
        )

    def distribution_from_frozen_state(
        self, state: FrozenDualGraphState, *, temperature: float = 1.0
    ) -> DualGraphDistributionCache:
        return self._distribution(state, temperature)

    def distribution(
        self, env: HeterogeneousDagEnv, temperature: float = 1.0
    ) -> DualGraphDistributionCache:
        if self._scenario is not env.scenario:
            self.reset(env.scenario)
        assert self._graph is not None
        return self._distribution(
            freeze_dual_graph_state(env, graph=self._graph), temperature
        )

    def select_action(self, env: HeterogeneousDagEnv) -> tuple[int, int]:
        cache = self.distribution(env)
        index = (
            int(np.argmax(cache.probabilities))
            if self.deterministic
            else int(self.rng.choice(len(cache.actions), p=cache.probabilities))
        )
        return cache.actions[index]

    def empty_gradients(self) -> dict[str, np.ndarray]:
        return {name: np.zeros_like(value) for name, value in self.params.items()}

    @staticmethod
    def add_gradients(
        target: dict[str, np.ndarray],
        source: Mapping[str, np.ndarray],
        scale: float = 1.0,
    ) -> None:
        if set(target) != set(source):
            raise ValueError("gradient dictionaries have different parameters")
        for name in target:
            target[name] += source[name] * scale

    def log_probability_gradients(
        self, cache: DualGraphDistributionCache, selected_index: int
    ) -> dict[str, np.ndarray]:
        if not 0 <= selected_index < len(cache.probabilities):
            raise IndexError("selected_index is outside the masked distribution")
        gradient = -cache.probabilities.copy()
        gradient[selected_index] += 1.0
        return self.score_gradients(cache, gradient)

    def entropy_gradients(
        self, cache: DualGraphDistributionCache
    ) -> dict[str, np.ndarray]:
        probabilities = cache.probabilities
        logs = np.log(probabilities + 1e-12)
        entropy = -float(np.sum(probabilities * logs))
        return self.score_gradients(
            cache, -probabilities * (logs + entropy)
        )

    def score_gradients(
        self, cache: DualGraphDistributionCache, d_scores: np.ndarray
    ) -> dict[str, np.ndarray]:
        score_gradient = np.asarray(d_scores, dtype=np.float64).copy()
        if score_gradient.shape != cache.probabilities.shape:
            raise ValueError("score gradient shape does not match probabilities")
        score_gradient /= cache.temperature
        output_w = cache.hidden.T @ score_gradient
        pair_input_gradient = (
            score_gradient[:, None] * self.params["output_w"][None, :]
        ) * (1.0 - cache.hidden**2)
        gradients: dict[str, np.ndarray] = {
            "output_w": output_w,
            "pair_w": cache.pair_features.T @ pair_input_gradient,
            "task_context_w": cache.task_context[cache.task_indices].T
            @ pair_input_gradient,
            "resource_context_w": cache.resource_context[
                cache.resource_indices
            ].T
            @ pair_input_gradient,
            "pair_b": np.sum(pair_input_gradient, axis=0),
        }

        task_context_gradient = np.zeros_like(cache.task_context)
        np.add.at(
            task_context_gradient,
            cache.task_indices,
            pair_input_gradient @ self.params["task_context_w"].T,
        )
        task_message_gradient = task_context_gradient * (1.0 - cache.task_context**2)
        gradients.update(
            {
                "task_self_w": cache.task_hidden.T @ task_message_gradient,
                "task_predecessor_w": cache.predecessor_hidden.T
                @ task_message_gradient,
                "task_successor_w": cache.successor_hidden.T
                @ task_message_gradient,
                "task_message_b": np.sum(task_message_gradient, axis=0),
            }
        )
        task_hidden_gradient = (
            task_message_gradient @ self.params["task_self_w"].T
            + cache.predecessor_adjacency.T
            @ (task_message_gradient @ self.params["task_predecessor_w"].T)
            + cache.successor_adjacency.T
            @ (task_message_gradient @ self.params["task_successor_w"].T)
        )
        task_node_gradient = task_hidden_gradient * (1.0 - cache.task_hidden**2)
        gradients["task_node_w"] = cache.task_features.T @ task_node_gradient
        gradients["task_node_b"] = np.sum(task_node_gradient, axis=0)

        resource_context_gradient = np.zeros_like(cache.resource_context)
        np.add.at(
            resource_context_gradient,
            cache.resource_indices,
            pair_input_gradient @ self.params["resource_context_w"].T,
        )
        resource_message_gradient = resource_context_gradient * (
            1.0 - cache.resource_context**2
        )
        gradients.update(
            {
                "resource_self_w": cache.resource_hidden.T
                @ resource_message_gradient,
                "resource_neighbor_w": cache.resource_neighbor_hidden.T
                @ resource_message_gradient,
                "resource_message_b": np.sum(resource_message_gradient, axis=0),
            }
        )
        resource_hidden_gradient = (
            resource_message_gradient @ self.params["resource_self_w"].T
            + cache.resource_adjacency.T
            @ (resource_message_gradient @ self.params["resource_neighbor_w"].T)
        )
        resource_node_gradient = resource_hidden_gradient * (
            1.0 - cache.resource_hidden**2
        )
        gradients["resource_node_w"] = (
            cache.resource_features.T @ resource_node_gradient
        )
        gradients["resource_node_b"] = np.sum(resource_node_gradient, axis=0)
        return gradients

    def apply_gradients(
        self,
        gradients: Mapping[str, np.ndarray],
        learning_rate: float,
        clip_norm: float,
    ) -> float:
        if learning_rate <= 0 or set(gradients) != set(self.params):
            raise ValueError("invalid Actor gradient update")
        checked = {
            name: np.asarray(gradients[name], dtype=np.float64)
            for name in self.params
        }
        if any(
            checked[name].shape != self.params[name].shape
            or not np.all(np.isfinite(checked[name]))
            for name in checked
        ):
            raise ValueError("Actor gradient contains an invalid value")
        norm = float(
            np.sqrt(sum(float(np.sum(value * value)) for value in checked.values()))
        )
        if norm > clip_norm > 0:
            factor = clip_norm / (norm + 1e-12)
            checked = {name: value * factor for name, value in checked.items()}
        self._adam_step += 1
        for name, gradient in checked.items():
            self._adam_m[name] = 0.9 * self._adam_m[name] + 0.1 * gradient
            self._adam_v[name] = 0.999 * self._adam_v[name] + 0.001 * gradient**2
            m_hat = self._adam_m[name] / (1.0 - 0.9**self._adam_step)
            v_hat = self._adam_v[name] / (1.0 - 0.999**self._adam_step)
            self.params[name] += learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
        return norm

    def clone(
        self, *, deterministic: bool = True, include_optimizer: bool = True
    ) -> "CloudEdgeDualGraphPolicy":
        clone = CloudEdgeDualGraphPolicy(
            self.hidden_dim,
            self.task_message_dim,
            self.resource_message_dim,
            self.seed,
            deterministic,
        )
        for name in self.params:
            clone.params[name] = self.params[name].copy()
            if include_optimizer:
                clone._adam_m[name] = self._adam_m[name].copy()
                clone._adam_v[name] = self._adam_v[name].copy()
        if include_optimizer:
            clone._adam_step = self._adam_step
            clone.rng.bit_generator.state = copy.deepcopy(self.rng.bit_generator.state)
        return clone

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            architecture=np.asarray(self.architecture),
            hidden_dim=np.asarray([self.hidden_dim]),
            task_message_dim=np.asarray([self.task_message_dim]),
            resource_message_dim=np.asarray([self.resource_message_dim]),
            seed=np.asarray([self.seed]),
            pair_feature_names=np.asarray(DUAL_GRAPH_PAIR_FEATURE_NAMES),
            task_node_feature_names=np.asarray(TASK_NODE_FEATURE_NAMES),
            resource_node_feature_names=np.asarray(RESOURCE_NODE_FEATURE_NAMES),
            **self.params,
        )

    @classmethod
    def load(
        cls, path: str | Path, deterministic: bool = True
    ) -> "CloudEdgeDualGraphPolicy":
        with np.load(path, allow_pickle=False) as data:
            if str(data["architecture"].item()) != cls.architecture:
                raise ValueError("unsupported dual-graph checkpoint architecture")
            policy = cls(
                int(data["hidden_dim"][0]),
                int(data["task_message_dim"][0]),
                int(data["resource_message_dim"][0]),
                int(data["seed"][0]),
                deterministic,
            )
            schemas = (
                ("pair_feature_names", DUAL_GRAPH_PAIR_FEATURE_NAMES),
                ("task_node_feature_names", TASK_NODE_FEATURE_NAMES),
                ("resource_node_feature_names", RESOURCE_NODE_FEATURE_NAMES),
            )
            for key, expected in schemas:
                if tuple(str(item) for item in data[key].tolist()) != expected:
                    raise ValueError(f"dual-graph checkpoint {key} changed")
            for name, expected in policy.params.items():
                value = np.asarray(data[name], dtype=np.float64)
                if value.shape != expected.shape or not np.all(np.isfinite(value)):
                    raise ValueError(f"invalid dual-graph parameter {name!r}")
                policy.params[name] = value.copy()
        return policy


@dataclass(frozen=True)
class DualGraphPPOTransition:
    state: FrozenDualGraphState
    selected_index: int
    old_log_probability: float
    value_state: np.ndarray
    advantage: float
    return_value: float


def _collect_episode(
    actor: CloudEdgeDualGraphPolicy,
    critic: Any,
    scenario: Scenario,
    heft_makespan: float,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[list[DualGraphPPOTransition], float]:
    from .ppo import compute_gae

    env = HeterogeneousDagEnv(scenario)
    graph = freeze_dual_graph(scenario)
    states: list[FrozenDualGraphState] = []
    selected: list[int] = []
    log_probabilities: list[float] = []
    value_states: list[np.ndarray] = []
    values: list[float] = []
    rewards: list[float] = []
    current_makespan = 0.0
    while not env.done:
        state = freeze_dual_graph_state(env, graph=graph)
        cache = actor.distribution_from_frozen_state(state)
        index = int(actor.rng.choice(len(cache.actions), p=cache.probabilities))
        value_state = critic.state_features(cache.pair_features)
        value = critic.predict(value_state)
        env.step(*cache.actions[index])
        next_makespan = max(item.finish for item in env.entries.values())
        rewards.append(-(next_makespan - current_makespan) / heft_makespan) # 计算奖励
        current_makespan = next_makespan
        states.append(state)
        selected.append(index)
        log_probabilities.append(float(np.log(cache.probabilities[index] + 1e-12)))
        value_states.append(value_state)
        values.append(value)
    advantages, returns = compute_gae(
        rewards, values, gamma=gamma, gae_lambda=gae_lambda
    )
    transitions = [
        DualGraphPPOTransition(
            states[index],
            selected[index],
            log_probabilities[index],
            value_states[index],
            float(advantages[index]),
            float(returns[index]),
        )
        for index in range(len(states))
    ]
    return transitions, current_makespan / heft_makespan  #  判断当前策略是否优于HEFT


def behavior_clone_dual_graph(
    actor: CloudEdgeDualGraphPolicy,
    scenarios: Sequence[Scenario],
    *,
    epochs: int = 5,
    learning_rate: float = 0.003,
    minibatch_size: int = 128,
    gradient_clip: float = 1.0,
    seed: int = 0,
) -> list[dict[str, float]]:
    """Warm-start the masked Actor from legal HEFT demonstrations."""

    if not scenarios or epochs < 0 or minibatch_size <= 0:
        raise ValueError("invalid behavior-cloning configuration")
    rng = np.random.default_rng(seed + 71)
    graphs = {id(scenario): freeze_dual_graph(scenario) for scenario in scenarios}
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        losses: list[float] = []
        correct = 0
        sample_count = 0
        gradients = actor.empty_gradients()
        batch_count = 0
        for raw_index in rng.permutation(len(scenarios)):
            scenario = scenarios[int(raw_index)]
            env = HeterogeneousDagEnv(scenario)
            teacher = HeftPolicy()
            teacher.reset(scenario)
            graph = graphs[id(scenario)]
            while not env.done:
                state = freeze_dual_graph_state(env, graph=graph)
                cache = actor.distribution_from_frozen_state(state)
                target = teacher.select_action(env)
                target_index = cache.actions.index(target)
                probability = float(cache.probabilities[target_index])
                losses.append(-float(np.log(probability + 1e-12)))
                correct += int(int(np.argmax(cache.probabilities)) == target_index)
                sample_count += 1
                actor.add_gradients(
                    gradients,
                    actor.log_probability_gradients(cache, target_index),
                )
                batch_count += 1
                env.step(*target)
                if batch_count >= minibatch_size:
                    actor.apply_gradients(
                        {
                            name: value / batch_count
                            for name, value in gradients.items()
                        },
                        learning_rate,
                        gradient_clip,
                    )
                    gradients = actor.empty_gradients()
                    batch_count = 0
        if batch_count:
            actor.apply_gradients(
                {name: value / batch_count for name, value in gradients.items()},
                learning_rate,
                gradient_clip,
            )
        history.append(
            {
                "epoch": float(epoch),
                "cross_entropy": float(np.mean(losses)),
                "teacher_action_accuracy": correct / sample_count,
                "sample_count": float(sample_count),
            }
        )
    return history


def _mean_ratio(
    actor: CloudEdgeDualGraphPolicy,
    scenarios: Sequence[Scenario],
    references: Mapping[str, float],
) -> float:
    evaluation_policy = actor.clone(deterministic=True, include_optimizer=False)
    return float(
        np.mean(
            [
                run_policy(scenario, evaluation_policy).makespan
                / references[scenario.id]
                for scenario in scenarios
            ]
        )
    )


def train_dual_graph_ppo(
    scenarios: Sequence[Scenario],
    *,
    validation_scenarios: Sequence[Scenario] | None = None,
    epochs: int = 5,
    episodes_per_epoch: int | None = None,
    hidden_dim: int = 32,
    task_message_dim: int = 8,
    resource_message_dim: int = 8,
    value_hidden_dim: int = 32,
    imitation_epochs: int = 5,
    imitation_learning_rate: float = 0.003,
    actor_learning_rate: float = 3e-4,
    value_learning_rate: float = 1e-3,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    clip_ratio: float = 0.2,
    entropy_coefficient: float = 0.01,
    update_epochs: int = 4,
    minibatch_size: int = 64,
    gradient_clip: float = 0.5,
    seed: int = 0,
) -> tuple[CloudEdgeDualGraphPolicy, Any, list[dict[str, Any]]]:
    """Train the dual-graph Actor with masked PPO, Critic and GAE.

    This compact public API is deliberately dataset-agnostic: callers can mix
    generated scenarios and converted external datasets in the same sequence.
    """

    from .ppo import ValueNetwork

    if not scenarios or epochs <= 0:
        raise ValueError("training requires scenarios and positive epochs")
    episode_count = len(scenarios) if episodes_per_epoch is None else episodes_per_epoch
    if not 1 <= episode_count <= len(scenarios):
        raise ValueError("episodes_per_epoch is outside the scenario set")
    if abs(gamma - 1.0) > 1e-12:
        raise ValueError("gamma must be 1.0 for the makespan reward identity")
    actor = CloudEdgeDualGraphPolicy(
        hidden_dim, task_message_dim, resource_message_dim, seed, False
    )
    critic = ValueNetwork(
        len(DUAL_GRAPH_PAIR_FEATURE_NAMES), value_hidden_dim, seed + 404
    )
    rng = np.random.default_rng(seed + 101)
    references = {
        scenario.id: run_policy(scenario, HeftPolicy()).makespan
        for scenario in scenarios
    }
    selection_scenarios = (
        list(validation_scenarios) if validation_scenarios is not None else list(scenarios)
    )
    if not selection_scenarios:
        raise ValueError("validation_scenarios must not be empty")
    selection_references = {
        scenario.id: run_policy(scenario, HeftPolicy()).makespan
        for scenario in selection_scenarios
    }
    imitation_history = behavior_clone_dual_graph(
        actor,
        scenarios,
        epochs=imitation_epochs,
        learning_rate=imitation_learning_rate,
        minibatch_size=minibatch_size,
        gradient_clip=gradient_clip,
        seed=seed,
    )
    best_actor = actor.clone(deterministic=True, include_optimizer=False)
    best_selection_ratio = _mean_ratio(
        actor, selection_scenarios, selection_references
    )
    best_epoch = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        transitions: list[DualGraphPPOTransition] = []
        ratios: list[float] = []
        for raw_index in rng.permutation(len(scenarios))[:episode_count]:
            scenario = scenarios[int(raw_index)]
            episode, ratio = _collect_episode(
                actor,
                critic,
                scenario,
                references[scenario.id],
                gamma=gamma,
                gae_lambda=gae_lambda,
            )
            transitions.extend(episode)
            ratios.append(ratio)
        raw_advantages = np.asarray([item.advantage for item in transitions])
        advantages = (raw_advantages - np.mean(raw_advantages)) / (
            np.std(raw_advantages) + 1e-8
        )
        updates: list[dict[str, float]] = []
        for _ in range(update_epochs):
            policy_losses: list[float] = []
            value_losses: list[float] = []
            permutation = rng.permutation(len(transitions))
            for start in range(0, len(transitions), minibatch_size):
                indices = permutation[start : start + minibatch_size]
                if len(indices) == 0:
                    continue
                actor_gradients = actor.empty_gradients()
                critic_gradients = critic.empty_gradients()
                for raw_index in indices:
                    index = int(raw_index)
                    item = transitions[index]
                    cache = actor.distribution_from_frozen_state(item.state)
                    new_log = float(
                        np.log(cache.probabilities[item.selected_index] + 1e-12)
                    )
                    ratio = float(np.exp(np.clip(new_log - item.old_log_probability, -20, 20)))
                    advantage = float(advantages[index])
                    clipped_ratio = float(
                        np.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
                    )
                    policy_losses.append(-min(ratio * advantage, clipped_ratio * advantage))
                    if ratio * advantage <= clipped_ratio * advantage + 1e-15:
                        actor.add_gradients(
                            actor_gradients,
                            actor.log_probability_gradients(cache, item.selected_index),
                            ratio * advantage,
                        )
                    if entropy_coefficient:
                        actor.add_gradients(
                            actor_gradients,
                            actor.entropy_gradients(cache),
                            entropy_coefficient,
                        )
                    loss, _, gradient = critic.loss_gradients(
                        item.value_state, item.return_value
                    )
                    value_losses.append(loss)
                    critic.add_gradients(critic_gradients, gradient)
                scale = 1.0 / len(indices)
                actor.apply_gradients(
                    {name: value * scale for name, value in actor_gradients.items()},
                    actor_learning_rate,
                    gradient_clip,
                )
                critic.apply_gradients(
                    {name: value * scale for name, value in critic_gradients.items()},
                    value_learning_rate,
                    gradient_clip,
                )
            updates.append(
                {
                    "policy_loss": float(np.mean(policy_losses)),
                    "value_loss": float(np.mean(value_losses)),
                }
            )
        history.append(
            {
                "epoch": epoch,
                "episode_count": len(ratios),
                "transition_count": len(transitions),
                "train_mean_ratio": float(np.mean(ratios)),
                "selection_mean_ratio": _mean_ratio(
                    actor, selection_scenarios, selection_references
                ),
                "updates": updates,
            }
        )
        if history[-1]["selection_mean_ratio"] < best_selection_ratio:
            best_selection_ratio = float(history[-1]["selection_mean_ratio"])
            best_epoch = epoch
            best_actor = actor.clone(deterministic=True, include_optimizer=False)
    if history:
        history[0]["behavior_cloning_warm_start"] = imitation_history
        history[0]["selection"] = {
            "split": "validation" if validation_scenarios is not None else "train",
            "best_epoch": best_epoch,
            "best_mean_ratio": best_selection_ratio,
            "fallback_to_behavior_cloning": best_epoch == 0,
        }
    return best_actor, critic, history


def dual_graph_metadata(policy: CloudEdgeDualGraphPolicy) -> dict[str, Any]:
    return {
        "display_name": policy.display_name,
        "architecture": policy.architecture,
        "policy_family": "actor_critic_reinforcement_learning",
        "policy_optimization": "masked_clipped_ppo_with_gae",
        "task_graph": "directed_data_weighted_dag",
        "resource_graph": "bandwidth_latency_weighted_cloud_edge_device_graph",
        "pair_feature_names": list(DUAL_GRAPH_PAIR_FEATURE_NAMES),
        "task_node_feature_names": list(TASK_NODE_FEATURE_NAMES),
        "resource_node_feature_names": list(RESOURCE_NODE_FEATURE_NAMES),
        "parameter_count": policy.parameter_count,
    }


def run_complex_dual_graph_pipeline(
    config_path: str | Path = "configs/complex_dual_graph.json",
    output_dir: str | Path | None = None,
) -> Path:
    """One-command complex-scenario training, validation and result export."""

    source = Path(config_path)
    config = json.loads(source.read_text(encoding="utf-8-sig"))
    seed = int(config["seed"])
    dataset = config["dataset"]
    training = config["training"]
    destination = Path(output_dir or config.get("output_dir", "outputs/dual-graph"))
    destination.mkdir(parents=True, exist_ok=True)
    low, high = (int(value) for value in dataset["task_range"])
    mixed_manifest: dict[str, Any] | None = None
    evaluation_splits: dict[str, list[Scenario]]
    if dataset.get("mode") == "mixed_v1":
        from .mixed_dataset import build_mixed_splits

        evaluation_splits, mixed_manifest = build_mixed_splits(config, source.resolve().parent)
        train_scenarios = evaluation_splits["train"]
        validation_scenarios = evaluation_splits["id_validation"]
        cache_dir = destination / "dataset_cache"
        for split_name, scenarios in evaluation_splits.items():
            split_dir = cache_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for scenario in scenarios:
                scenario.save(split_dir / f"{scenario.id}.json")
        (destination / "dataset_manifest.json").write_text(
            json.dumps(mixed_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        chooser = np.random.default_rng(seed)

        def build(count: int, offset: int, prefix: str) -> list[Scenario]:
            return [
                generate_complex_scenario(
                    seed + offset + index * 9973,
                    task_count=int(chooser.integers(low, high + 1)),
                    resource_count=int(dataset["resource_count"]),
                    edge_probability=float(dataset["edge_probability"]),
                    scenario_id=f"{prefix}-{index:04d}",
                )
                for index in range(count)
            ]

        train_scenarios = build(int(dataset["train_count"]), 10_000, "complex-train")
        validation_scenarios = build(
            int(dataset["validation_count"]), 900_000, "complex-validation"
        )
        evaluation_splits = {"id_validation": validation_scenarios}
    actor, critic, history = train_dual_graph_ppo(
        train_scenarios,
        validation_scenarios=validation_scenarios,
        epochs=int(training["epochs"]),
        episodes_per_epoch=int(training["episodes_per_epoch"]),
        hidden_dim=int(training["hidden_dim"]),
        task_message_dim=int(training["task_message_dim"]),
        resource_message_dim=int(training["resource_message_dim"]),
        value_hidden_dim=int(training["value_hidden_dim"]),
        imitation_epochs=int(training["imitation_epochs"]),
        imitation_learning_rate=float(training["imitation_learning_rate"]),
        actor_learning_rate=float(training["actor_learning_rate"]),
        value_learning_rate=float(training["value_learning_rate"]),
        gae_lambda=float(training["gae_lambda"]),
        clip_ratio=float(training["clip_ratio"]),
        entropy_coefficient=float(training["entropy_coefficient"]),
        update_epochs=int(training["update_epochs"]),
        minibatch_size=int(training["minibatch_size"]),
        gradient_clip=float(training["gradient_clip"]),
        seed=seed,
    )
    actor.save(destination / "actor.npz")
    critic.save(destination / "critic.npz")
    (destination / "training_curve.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    schedulers = (
        HeftPolicy(),
        CpopPolicy(),
        GreedyEarliestFinishPolicy(),
        actor,
    )
    rows: list[dict[str, Any]] = []
    split_metrics: dict[str, dict[str, Any]] = {}
    for split_name, scenarios in evaluation_splits.items():
        if split_name == "train" or not scenarios:
            continue
        ratios: dict[str, list[float]] = {policy.name: [] for policy in schedulers}
        for scenario in scenarios:
            results = {
                policy.name: run_policy(scenario, policy) for policy in schedulers
            }
            reference = results["heft"].makespan
            for policy in schedulers:
                result = results[policy.name]
                ratio = result.makespan / reference
                ratios[policy.name].append(ratio)
                rows.append(
                    {
                        "split": split_name,
                        "scenario_id": scenario.id,
                        "task_count": scenario.task_count,
                        "resource_count": scenario.resource_count,
                        "policy": policy.name,
                        "makespan": result.makespan,
                        "heft_makespan": reference,
                        "ratio_to_heft": ratio,
                    }
                )
        split_metrics[split_name] = {
            name: {
                "sample_count": len(values),
                "mean_ratio": float(np.mean(values)),
                "population_std_ratio": float(np.std(values)),
                "min_ratio": float(np.min(values)),
                "max_ratio": float(np.max(values)),
            }
            for name, values in ratios.items()
        }
    with (destination / "validation_per_instance.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = split_metrics["id_validation"]
    summary = {
        "format_version": 1,
        "method": dual_graph_metadata(actor),
        "competition_scope": {
            "resource_kinds": ["device", "edge", "cloud"],
            "objective": "makespan",
            "legal_action": "ready task x compatible resource",
            "timeline": "insertion_based_earliest_gap",
        },
        "dataset": (
            {
                "source": "stg+dagbench+synthetic_tasks__topology_zoo+dagbench+synthetic_networks",
                "adapter_version": mixed_manifest["adapter_version"],
                "counts": {name: len(items) for name, items in evaluation_splits.items()},
                "discovered": mixed_manifest["discovered"],
                "manifest": "dataset_manifest.json",
            }
            if mixed_manifest is not None
            else {
                "source": "generated_complex_cloud_edge_device_v1",
                "train_count": len(train_scenarios),
                "validation_count": len(validation_scenarios),
                "task_range": [low, high],
                "resource_count": int(dataset["resource_count"]),
            }
        ),
        "training": {
            "actor": "dual_graph_masked_policy",
            "critic": "mean_max_pooled_value_network",
            "advantage": "GAE",
            "optimizer": "clipped_PPO",
            "history": history,
        },
        "validation": metrics,
        "evaluation_splits": split_metrics,
        "primary_metric": {
            "name": "mean(RL_makespan / HEFT_makespan)",
            "value": metrics[actor.name]["mean_ratio"],
            "sample_count": metrics[actor.name]["sample_count"],
            "population_std": metrics[actor.name]["population_std_ratio"],
        },
        "artifacts": [
            "actor.npz",
            "critic.npz",
            "training_curve.json",
            "validation_per_instance.csv",
            "summary.json",
        ] + (["dataset_manifest.json", "dataset_cache/"] if mixed_manifest is not None else []),
    }
    summary_path = destination / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary_path
