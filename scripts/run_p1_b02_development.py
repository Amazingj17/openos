from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from trisched.bc import policy_parameter_hash
from trisched.gnn import (
    TASK_GNN_FEATURE_NAMES,
    TaskGNNPolicy,
    task_gnn_parameter_hash,
)
from trisched.learning import (
    FEATURE_NAMES,
    TEACHER_FEATURE_NAMES,
    MaskedMLPPolicy,
)
from trisched.ood import produce_development_evidence
from trisched.policies import (
    CpopPolicy,
    GreedyEarliestFinishPolicy,
    HeftPolicy,
    RandomPolicy,
)
from trisched.reporting import load_evaluation_contract
from trisched.schedulers import PolicySchedulerRunner, SchedulerRunner


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_path(path: Path, repository: Path) -> str:
    try:
        return path.resolve().relative_to(repository.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"checkpoint escapes repository: {path}") from error


def _checkpoint_metadata(
    path: Path,
    policy: MaskedMLPPolicy | TaskGNNPolicy,
    repository: Path,
) -> dict[str, Any]:
    parameter_sha256 = (
        task_gnn_parameter_hash(policy)
        if isinstance(policy, TaskGNNPolicy)
        else policy_parameter_hash(policy)
    )
    return {
        "path": _repository_path(path, repository),
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
        "parameter_sha256": parameter_sha256,
        "internal_seed": policy.seed,
        "feature_names": list(policy.feature_names),
    }


def build_runner_bundle(
    contract: Mapping[str, Any],
    *,
    repository: Path,
    bc_checkpoint: Path,
    masked_mlp_dir: Path,
    task_gnn_dir: Path,
) -> tuple[
    dict[tuple[str, int], SchedulerRunner],
    dict[str, Any],
]:
    expected_mlp_features = tuple(
        name for name in FEATURE_NAMES if name not in TEACHER_FEATURE_NAMES
    )
    runners: dict[tuple[str, int], SchedulerRunner] = {}
    checkpoints: dict[str, dict[str, Any]] = {
        "bc": {},
        "masked_mlp": {},
        "task_gnn": {},
    }
    policies = {
        item["id"]: list(item["required_seeds"]) for item in contract["policies"]
    }

    for seed in policies["heft"]:
        runners[("heft", seed)] = PolicySchedulerRunner("heft", HeftPolicy)
    for seed in policies["greedy_eft"]:
        runners[("greedy_eft", seed)] = PolicySchedulerRunner(
            "greedy_eft",
            GreedyEarliestFinishPolicy,
        )
    for seed in policies["cpop"]:
        runners[("cpop", seed)] = PolicySchedulerRunner("cpop", CpopPolicy)
    for seed in policies["random"]:
        runners[("random", seed)] = PolicySchedulerRunner(
            "random",
            lambda seed=seed: RandomPolicy(seed=seed),
        )

    bc_seeds = policies["bc"]
    if len(bc_seeds) != 1:
        raise ValueError("BC baseline must have exactly one frozen seed")
    bc_seed = bc_seeds[0]
    bc_policy = MaskedMLPPolicy.load(bc_checkpoint)
    if bc_policy.seed != bc_seed:
        raise ValueError("BC checkpoint seed does not match the contract")
    if tuple(bc_policy.feature_names) != FEATURE_NAMES:
        raise ValueError("BC checkpoint does not use the frozen 16-D schema")
    bc_policy.name = "bc"
    runners[("bc", bc_seed)] = PolicySchedulerRunner("bc", lambda: bc_policy)
    checkpoints["bc"][str(bc_seed)] = _checkpoint_metadata(
        bc_checkpoint,
        bc_policy,
        repository,
    )

    for seed in policies["masked_mlp"]:
        path = masked_mlp_dir / f"seed_{seed}_ppo_best_policy.npz"
        policy = MaskedMLPPolicy.load(path)
        if policy.seed != seed:
            raise ValueError(f"masked MLP checkpoint seed mismatch: {seed}")
        if tuple(policy.feature_names) != expected_mlp_features:
            raise ValueError(f"masked MLP feature schema mismatch: {seed}")
        runners[("masked_mlp", seed)] = PolicySchedulerRunner(
            "masked_mlp",
            lambda policy=policy: policy,
        )
        checkpoints["masked_mlp"][str(seed)] = _checkpoint_metadata(
            path,
            policy,
            repository,
        )

    for seed in policies["task_gnn"]:
        path = task_gnn_dir / f"seed_{seed}_task_gnn_ppo_best_policy.npz"
        policy = TaskGNNPolicy.load(path)
        if policy.seed != seed:
            raise ValueError(f"task-GNN checkpoint seed mismatch: {seed}")
        if tuple(policy.feature_names) != TASK_GNN_FEATURE_NAMES:
            raise ValueError(f"task-GNN feature schema mismatch: {seed}")
        runners[("task_gnn", seed)] = PolicySchedulerRunner(
            "task_gnn",
            lambda policy=policy: policy,
        )
        checkpoints["task_gnn"][str(seed)] = _checkpoint_metadata(
            path,
            policy,
            repository,
        )

    expected = {
        (item["id"], seed)
        for item in contract["policies"]
        for seed in item["required_seeds"]
    }
    if set(runners) != expected:
        raise ValueError("runner bundle does not match the frozen policy/seed grid")
    return runners, {
        "format_version": 1,
        "checkpoints": checkpoints,
        "baselines": {
            "heft": "trisched.policies.HeftPolicy",
            "greedy_eft": "trisched.policies.GreedyEarliestFinishPolicy",
            "cpop": "trisched.policies.CpopPolicy",
            "random": "trisched.policies.RandomPolicy",
        },
    }


def _git_code_metadata(repository: Path) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        text=True,
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repository,
            text=True,
        ).strip()
    )
    if dirty:
        raise RuntimeError("formal development evidence requires a clean worktree")
    return {
        "commit": commit,
        "working_tree_dirty": False,
        "source": "git",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run frozen P1-B02 development ID/OOD evidence",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=REPOSITORY / "configs" / "p1_b02_evaluation_contract.json",
    )
    parser.add_argument(
        "--materialization-root",
        type=Path,
        default=REPOSITORY / "outputs" / "p1-b02-development-slices-v2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY
        / "outputs"
        / "p1-b02-development-evidence"
        / "development-evidence.json",
    )
    parser.add_argument(
        "--bc-checkpoint",
        type=Path,
        default=REPOSITORY / "outputs" / "p1-a01-stg-bc" / "bc_best.npz",
    )
    parser.add_argument(
        "--masked-mlp-dir",
        type=Path,
        default=REPOSITORY / "outputs" / "p1-a04-stg-ppo-5seed",
    )
    parser.add_argument(
        "--task-gnn-dir",
        type=Path,
        default=REPOSITORY / "outputs" / "p1-a03-stg-task-gnn",
    )
    args = parser.parse_args(argv)

    contract = load_evaluation_contract(args.contract)
    runners, bundle = build_runner_bundle(
        contract,
        repository=REPOSITORY,
        bc_checkpoint=args.bc_checkpoint,
        masked_mlp_dir=args.masked_mlp_dir,
        task_gnn_dir=args.task_gnn_dir,
    )
    code = _git_code_metadata(REPOSITORY)
    code["runner_bundle"] = {
        **bundle,
        "script": {
            "path": _repository_path(Path(__file__), REPOSITORY),
            "sha256": _file_sha256(Path(__file__)),
        },
    }

    def provider(policy: str, seed: int) -> SchedulerRunner:
        return runners[(policy, seed)]

    evidence = produce_development_evidence(
        args.contract,
        args.materialization_root,
        provider,
        args.output,
        code=code,
    )
    print(evidence.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
