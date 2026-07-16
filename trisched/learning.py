from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .env import HeterogeneousDagEnv, ScheduleResult, run_policy, validate_schedule
from .policies import HeftPolicy, compute_upward_ranks
from .scenario import Scenario


FEATURE_NAMES = (
    "task_workload",
    "execution_time",
    "resource_speed",
    "earliest_start",
    "earliest_finish",
    "resource_ready",
    "communication_delay",
    "upward_rank",
    "indegree",
    "outdegree",
    "progress",
    "is_device",
    "is_edge",
    "is_cloud",
    "is_heft_task",
    "is_heft_pair",
)


def candidate_features(
    env: HeterogeneousDagEnv, ranks: np.ndarray
) -> tuple[tuple[tuple[int, int], ...], np.ndarray]:
    scenario = env.scenario
    candidates = env.candidate_actions()
    max_workload = max(task.workload for task in scenario.tasks)
    max_speed = max(resource.speed for resource in scenario.resources)
    max_execution = max(
        scenario.execution_time(task.id, resource.id)
        for task in scenario.tasks
        for resource in scenario.resources
    )
    total_work = sum(task.workload for task in scenario.tasks)
    min_speed = min(resource.speed for resource in scenario.resources)
    time_scale = max(total_work / min_speed, 1.0)
    max_rank = max(float(np.max(ranks)), 1.0)
    predecessors = scenario.predecessors()
    successors = scenario.successors()
    max_degree = max(
        1,
        max(
            max(len(predecessors[i]), len(successors[i]))
            for i in range(scenario.task_count)
        ),
    )
    ready_tasks = env.ready_tasks()
    heft_task = min(ready_tasks, key=lambda item: (-ranks[item], item))
    heft_resource = min(
        range(scenario.resource_count),
        key=lambda item: (env.earliest_slot(heft_task, item)[1], item),
    )
    rows: list[list[float]] = []
    for task_id, resource_id in candidates:
        task = scenario.tasks[task_id]
        resource = scenario.resources[resource_id]
        start, finish = env.earliest_slot(task_id, resource_id)
        dependency_ready = env.dependency_ready_time(task_id, resource_id)
        parent_finish = max(
            (env.entries[parent].finish for parent in predecessors[task_id]),
            default=0.0,
        )
        communication_delay = max(0.0, dependency_ready - parent_finish)
        rows.append(
            [
                task.workload / max_workload,
                scenario.execution_time(task_id, resource_id) / max_execution,
                resource.speed / max_speed,
                start / time_scale,
                finish / time_scale,
                env.resource_ready_time(resource_id) / time_scale,
                communication_delay / time_scale,
                ranks[task_id] / max_rank,
                len(predecessors[task_id]) / max_degree,
                len(successors[task_id]) / max_degree,
                env.progress,
                1.0 if resource.kind == "device" else 0.0,
                1.0 if resource.kind == "edge" else 0.0,
                1.0 if resource.kind == "cloud" else 0.0,
                1.0 if task_id == heft_task else 0.0,
                1.0 if (task_id, resource_id) == (heft_task, heft_resource) else 0.0,
            ]
        )
    return candidates, np.asarray(rows, dtype=np.float64)


@dataclass
class DistributionCache:
    actions: tuple[tuple[int, int], ...]
    features: np.ndarray
    hidden: np.ndarray
    probabilities: np.ndarray
    temperature: float


class MaskedMLPPolicy:
    """A variable-action neural pair scorer with legal candidates as its mask."""

    name = "masked_mlp"

    def __init__(
        self,
        hidden_dim: int = 32,
        seed: int = 0,
        deterministic: bool = True,
    ) -> None:
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.hidden_dim = hidden_dim
        self.seed = seed
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)
        input_dim = len(FEATURE_NAMES)
        limit_1 = np.sqrt(6.0 / (input_dim + hidden_dim))
        limit_2 = np.sqrt(6.0 / (hidden_dim + 1))
        self.params: dict[str, np.ndarray] = {
            "w1": self.rng.uniform(-limit_1, limit_1, (input_dim, hidden_dim)),
            "b1": np.zeros(hidden_dim, dtype=np.float64),
            "w2": self.rng.uniform(-limit_2, limit_2, hidden_dim),
        }
        self._adam_m = {name: np.zeros_like(value) for name, value in self.params.items()}
        self._adam_v = {name: np.zeros_like(value) for name, value in self.params.items()}
        self._adam_step = 0
        self.ranks: np.ndarray | None = None

    def reset(self, scenario: Scenario) -> None:
        self.ranks = compute_upward_ranks(scenario)

    def distribution(
        self, env: HeterogeneousDagEnv, temperature: float = 1.0
    ) -> DistributionCache:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.ranks is None:
            self.reset(env.scenario)
        assert self.ranks is not None
        actions, features = candidate_features(env, self.ranks)
        hidden = np.tanh(features @ self.params["w1"] + self.params["b1"])
        scores = (hidden @ self.params["w2"]) / temperature
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores)
        probabilities = exp_scores / np.sum(exp_scores)
        return DistributionCache(actions, features, hidden, probabilities, temperature)

    def select_action(self, env: HeterogeneousDagEnv) -> tuple[int, int]:
        cache = self.distribution(env)
        if self.deterministic:
            index = int(np.argmax(cache.probabilities))
        else:
            index = int(self.rng.choice(len(cache.actions), p=cache.probabilities))
        return cache.actions[index]

    def sample(
        self, env: HeterogeneousDagEnv, temperature: float = 1.0
    ) -> tuple[tuple[int, int], int, DistributionCache]:
        cache = self.distribution(env, temperature=temperature)
        index = int(self.rng.choice(len(cache.actions), p=cache.probabilities))
        return cache.actions[index], index, cache

    def empty_gradients(self) -> dict[str, np.ndarray]:
        return {name: np.zeros_like(value) for name, value in self.params.items()}

    def log_probability_gradients(
        self, cache: DistributionCache, selected_index: int
    ) -> dict[str, np.ndarray]:
        d_scores = -cache.probabilities.copy()
        d_scores[selected_index] += 1.0
        d_scores /= cache.temperature
        hidden_gradient = (
            d_scores[:, None] * self.params["w2"][None, :]
        ) * (1.0 - cache.hidden**2)
        return {
            "w2": cache.hidden.T @ d_scores,
            "w1": cache.features.T @ hidden_gradient,
            "b1": np.sum(hidden_gradient, axis=0),
        }

    @staticmethod
    def add_gradients(
        target: dict[str, np.ndarray], source: dict[str, np.ndarray], scale: float = 1.0
    ) -> None:
        for name in target:
            target[name] += source[name] * scale

    def apply_gradients(
        self,
        gradients: dict[str, np.ndarray],
        learning_rate: float,
        clip_norm: float,
    ) -> float:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        norm = float(
            np.sqrt(sum(float(np.sum(value * value)) for value in gradients.values()))
        )
        if norm > clip_norm > 0:
            scale = clip_norm / (norm + 1e-12)
            gradients = {name: value * scale for name, value in gradients.items()}
        self._adam_step += 1
        beta1, beta2 = 0.9, 0.999
        for name, gradient in gradients.items():
            self._adam_m[name] = beta1 * self._adam_m[name] + (1 - beta1) * gradient
            self._adam_v[name] = beta2 * self._adam_v[name] + (1 - beta2) * (
                gradient * gradient
            )
            m_hat = self._adam_m[name] / (1 - beta1**self._adam_step)
            v_hat = self._adam_v[name] / (1 - beta2**self._adam_step)
            # Gradients are for log-probability, so Adam performs gradient ascent.
            self.params[name] += learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
        return norm

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            hidden_dim=np.asarray([self.hidden_dim], dtype=np.int64),
            seed=np.asarray([self.seed], dtype=np.int64),
            feature_names=np.asarray(FEATURE_NAMES),
            **self.params,
        )

    @classmethod
    def load(
        cls, path: str | Path, deterministic: bool = True
    ) -> "MaskedMLPPolicy":
        data = np.load(path, allow_pickle=False)
        stored_features = tuple(str(item) for item in data["feature_names"].tolist())
        if stored_features != FEATURE_NAMES:
            raise ValueError(
                "checkpoint feature schema does not match this TriSched version"
            )
        policy = cls(
            hidden_dim=int(data["hidden_dim"][0]),
            seed=int(data["seed"][0]),
            deterministic=deterministic,
        )
        for name in policy.params:
            policy.params[name] = np.asarray(data[name], dtype=np.float64)
        return policy


def _teacher_episode(
    policy: MaskedMLPPolicy,
    scenario: Scenario,
    learning_rate: float,
    gradient_clip: float,
) -> tuple[float, float]:
    env = HeterogeneousDagEnv(scenario)
    teacher = HeftPolicy()
    teacher.reset(scenario)
    policy.reset(scenario)
    total_loss = 0.0
    correct = 0
    steps = 0
    while not env.done:
        expert_action = teacher.select_action(env)
        cache = policy.distribution(env)
        target = cache.actions.index(expert_action)
        prediction = int(np.argmax(cache.probabilities))
        correct += int(prediction == target)
        total_loss -= float(np.log(cache.probabilities[target] + 1e-12))
        gradients = policy.log_probability_gradients(cache, target)
        policy.apply_gradients(gradients, learning_rate, gradient_clip)
        env.step(*expert_action)
        steps += 1
    return total_loss / steps, correct / steps


def train_imitation(
    policy: MaskedMLPPolicy,
    scenarios: list[Scenario],
    epochs: int,
    learning_rate: float,
    gradient_clip: float,
    seed: int,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        losses: list[float] = []
        accuracies: list[float] = []
        for index in rng.permutation(len(scenarios)):
            loss, accuracy = _teacher_episode(
                policy,
                scenarios[int(index)],
                learning_rate,
                gradient_clip,
            )
            losses.append(loss)
            accuracies.append(accuracy)
        history.append(
            {
                "epoch": float(epoch + 1),
                "mean_cross_entropy": float(np.mean(losses)),
                "mean_action_accuracy": float(np.mean(accuracies)),
            }
        )
    return history


def _reinforce_rollout(
    policy: MaskedMLPPolicy, scenario: Scenario, temperature: float
) -> tuple[ScheduleResult, dict[str, np.ndarray], int]:
    env = HeterogeneousDagEnv(scenario)
    policy.reset(scenario)
    gradients = policy.empty_gradients()
    steps = 0
    while not env.done:
        action, selected_index, cache = policy.sample(env, temperature=temperature)
        step_gradients = policy.log_probability_gradients(cache, selected_index)
        policy.add_gradients(gradients, step_gradients)
        env.step(*action)
        steps += 1
    result = env.result(policy.name)
    validate_schedule(scenario, result)
    return result, gradients, steps


def train_reinforce(
    policy: MaskedMLPPolicy,
    scenarios: list[Scenario],
    epochs: int,
    learning_rate: float,
    gradient_clip: float,
    seed: int,
    temperature: float,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    heft_makespans = {
        scenario.id: run_policy(scenario, HeftPolicy()).makespan
        for scenario in scenarios
    }
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        episodes: list[
            tuple[float, dict[str, np.ndarray], int]
        ] = []
        ratios: list[float] = []
        for index in rng.permutation(len(scenarios)):
            scenario = scenarios[int(index)]
            result, gradients, steps = _reinforce_rollout(
                policy, scenario, temperature=temperature
            )
            ratio = result.makespan / heft_makespans[scenario.id]
            reward = -ratio
            episodes.append((reward, gradients, steps))
            ratios.append(ratio)
        baseline = float(np.mean([episode[0] for episode in episodes]))
        advantages: list[float] = []
        norms: list[float] = []
        for reward, gradients, steps in episodes:
            advantage = reward - baseline
            advantages.append(advantage)
            scaled = {
                name: value * (advantage / max(steps, 1))
                for name, value in gradients.items()
            }
            norms.append(
                policy.apply_gradients(scaled, learning_rate, gradient_clip)
            )
        history.append(
            {
                "epoch": float(epoch + 1),
                "mean_ratio": float(np.mean(ratios)),
                "mean_reward": float(-np.mean(ratios)),
                "advantage_std": float(np.std(advantages)),
                "mean_gradient_norm": float(np.mean(norms)),
            }
        )
    return history


def train_policy(
    scenarios: list[Scenario],
    config: dict[str, Any],
    seed: int,
) -> tuple[MaskedMLPPolicy, dict[str, Any]]:
    policy = MaskedMLPPolicy(
        hidden_dim=int(config.get("hidden_dim", 32)),
        seed=seed,
        deterministic=True,
    )
    imitation = train_imitation(
        policy,
        scenarios,
        epochs=int(config.get("imitation_epochs", 8)),
        learning_rate=float(config.get("imitation_learning_rate", 0.008)),
        gradient_clip=float(config.get("gradient_clip", 5.0)),
        seed=seed + 101,
    )
    reinforce = train_reinforce(
        policy,
        scenarios,
        epochs=int(config.get("reinforce_epochs", 5)),
        learning_rate=float(config.get("reinforce_learning_rate", 0.002)),
        gradient_clip=float(config.get("gradient_clip", 5.0)),
        seed=seed + 202,
        temperature=float(config.get("reinforce_temperature", 2.0)),
    )
    return policy, {
        "algorithm": "HEFT imitation + episodic REINFORCE",
        "feature_names": list(FEATURE_NAMES),
        "imitation": imitation,
        "reinforce": reinforce,
    }
