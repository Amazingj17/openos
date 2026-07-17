from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .bc import BehaviorCloningError, run_bc_pipeline
from .evaluation import (
    DEFAULT_FAILURE_PENALTY_RATIO,
    dataset_manifest,
    evaluate_split,
    resolve_failure_penalty_ratio,
    write_summary,
)
from .learning import MaskedMLPPolicy, train_policy
from .ppo import run_ppo_pipeline, run_task_gnn_pipeline
from .reporting import (
    EvaluationReportError,
    build_evaluation_report,
    claim_public_test_gate,
)
from .scenario import Scenario, ScenarioValidationError, generate_dataset
from .schedulers import SchedulerAdapterError


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_metadata(repository_hint: Path | None = None) -> dict[str, Any]:
    environment_head = os.environ.get("TRISCHED_GIT_HEAD", "").strip()
    if environment_head:
        dirty_value = os.environ.get("TRISCHED_GIT_DIRTY", "").strip().lower()
        environment_dirty = {
            "true": True,
            "1": True,
            "false": False,
            "0": False,
        }.get(dirty_value)
        return {
            "commit": environment_head,
            "working_tree_dirty": environment_dirty,
            "source": "TRISCHED_GIT_HEAD",
        }

    candidates = []
    if repository_hint is not None:
        candidates.append(repository_hint.resolve())
    candidates.append(Path(__file__).resolve().parents[1])
    for repository in candidates:
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
            continue
        return {
            "commit": commit,
            "working_tree_dirty": bool(status.strip()),
            "source": "git",
        }
    return {
        "commit": None,
        "working_tree_dirty": None,
        "source": "unavailable",
    }


def write_run_manifest(
    output_dir: Path,
    *,
    mode: str,
    config_source: Path,
    config: dict[str, Any],
    dataset_manifests: dict[str, dict[str, Any]],
    checkpoint: Path,
    artifact_names: list[str],
    split: str | None = None,
) -> Path:
    """Write a non-circular manifest for one completed pipeline/evaluation run."""

    artifacts: dict[str, dict[str, Any]] = {}
    for name in artifact_names:
        path = output_dir / name
        if not path.is_file():
            raise RuntimeError(f"run artifact is missing: {name}")
        artifacts[Path(name).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }

    dataset_inputs = {
        name: {
            "count": manifest["count"],
            "scenario_hashes_sha256": _json_sha256(manifest["scenario_hashes"]),
        }
        for name, manifest in sorted(dataset_manifests.items())
    }
    lockfile_candidates = [
        parent / "requirements-lock.txt"
        for parent in (config_source.parent, *config_source.parents)
    ]
    lockfile_candidates.append(
        Path(__file__).resolve().parents[1] / "requirements-lock.txt"
    )
    lockfile = next(
        (candidate for candidate in lockfile_candidates if candidate.is_file()),
        None,
    )
    dependency_lock = (
        {"path": lockfile.name, "sha256": _file_sha256(lockfile)}
        if lockfile is not None
        else None
    )
    manifest = {
        "format_version": 1,
        "mode": mode,
        "split": split,
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "hash_algorithm": "sha256",
        "path_convention": "output-relative POSIX paths",
        "code": _git_metadata(config_source.parent),
        "runtime": {
            "trisched": __version__,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "inputs": {
            "config": {
                "source_name": config_source.name,
                "source_sha256": _file_sha256(config_source),
                "resolved_path": "resolved_config.json",
                "resolved_sha256": _file_sha256(output_dir / "resolved_config.json"),
            },
            "datasets": dataset_inputs,
            "dataset_manifest_sha256": _file_sha256(
                output_dir / "dataset_manifest.json"
            ),
            "checkpoint": {
                "name": checkpoint.name,
                "sha256": _file_sha256(checkpoint),
            },
            "dependency_lock": dependency_lock,
        },
        "scoring": {
            "failure_penalty_ratio": config["evaluation"]["failure_penalty_ratio"],
            "failures_remain_in_denominator": True,
            "publishable_requires_zero_failures": True,
        },
        "artifacts": dict(sorted(artifacts.items())),
    }
    path = output_dir / "run_manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    config = json.loads(source.read_text(encoding="utf-8"))
    required = {"seed", "output_dir", "dataset", "training", "evaluation"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"config is missing keys: {sorted(missing)}")
    dataset = config["dataset"]
    for key in ("train_count", "validation_count", "test_count"):
        if int(dataset.get(key, 0)) <= 0:
            raise ValueError(f"dataset.{key} must be positive")
    task_range = dataset.get("task_range")
    if not isinstance(task_range, list) or len(task_range) != 2:
        raise ValueError("dataset.task_range must be a two-element list")
    evaluation = config["evaluation"]
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation must be an object")
    evaluation["failure_penalty_ratio"] = resolve_failure_penalty_ratio(
        evaluation.get("failure_penalty_ratio", DEFAULT_FAILURE_PENALTY_RATIO)
    )
    return config


def build_splits(config: dict[str, Any]) -> dict[str, list[Scenario]]:
    seed = int(config["seed"])
    dataset = config["dataset"]
    common = {
        "task_range": tuple(int(x) for x in dataset["task_range"]),
        "resource_count": int(dataset.get("resource_count", 3)),
        "edge_probability": float(dataset.get("edge_probability", 0.18)),
    }
    splits = {
        "train": generate_dataset(
            int(dataset["train_count"]), seed + 10_000, prefix="train", **common
        ),
        "validation": generate_dataset(
            int(dataset["validation_count"]),
            seed + 20_000,
            prefix="validation",
            **common,
        ),
        "test": generate_dataset(
            int(dataset["test_count"]), seed + 30_000, prefix="test", **common
        ),
    }
    hashes = {
        name: {scenario.content_hash() for scenario in scenarios}
        for name, scenarios in splits.items()
    }
    if hashes["train"] & hashes["validation"]:
        raise RuntimeError("train and validation scenarios overlap")
    if hashes["train"] & hashes["test"]:
        raise RuntimeError("train and test scenarios overlap")
    if hashes["validation"] & hashes["test"]:
        raise RuntimeError("validation and test scenarios overlap")
    return splits


def run_pipeline(config_path: str | Path, output_override: str | None = None) -> Path:
    config_source = Path(config_path).resolve()
    config = load_config(config_source)
    if output_override:
        config["output_dir"] = output_override
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    started = time.perf_counter()
    print("[1/4] generating deterministic train/validation/test scenarios")
    splits = build_splits(config)
    manifests = {
        name: dataset_manifest(scenarios, name) for name, scenarios in splits.items()
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifests, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[2/4] training masked MLP with HEFT imitation and REINFORCE")
    policy, training_history = train_policy(
        splits["train"], config["training"], seed=int(config["seed"])
    )
    checkpoint = output_dir / "masked_mlp.npz"
    policy.save(checkpoint)
    (output_dir / "training_history.json").write_text(
        json.dumps(training_history, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[3/4] evaluating configured schedulers through one validated path")
    random_seed = int(config["evaluation"].get("random_seed", 991))
    failure_penalty_ratio = config["evaluation"]["failure_penalty_ratio"]
    validation_metrics, _ = evaluate_split(
        splits["validation"],
        policy,
        "validation",
        output_dir,
        random_seed,
        config["evaluation"].get("schedulers"),
        config_source.parent,
        failure_penalty_ratio,
    )
    test_metrics, _ = evaluate_split(
        splits["test"],
        policy,
        "test",
        output_dir,
        random_seed,
        config["evaluation"].get("schedulers"),
        config_source.parent,
        failure_penalty_ratio,
    )

    training_history["wall_clock_seconds"] = time.perf_counter() - started
    print("[4/4] writing summary.json and reproducibility metadata")
    summary_path = write_summary(
        output_dir,
        config,
        training_history,
        validation_metrics,
        test_metrics,
        manifests,
    )
    write_run_manifest(
        output_dir,
        mode="pipeline",
        config_source=config_source,
        config=config,
        dataset_manifests=manifests,
        checkpoint=checkpoint,
        artifact_names=[
            "resolved_config.json",
            "dataset_manifest.json",
            "training_history.json",
            "masked_mlp.npz",
            "validation_per_instance.csv",
            "validation_failures.jsonl",
            "validation_example_schedule.json",
            "test_per_instance.csv",
            "test_failures.jsonl",
            "test_example_schedule.json",
            "summary.json",
        ],
    )
    ratio = test_metrics["masked_mlp"]["mean_ratio"]
    print(f"done: test mean_ratio={ratio:.4f} (lower is better; HEFT=1.0)")
    print(f"summary: {summary_path.resolve()}")
    return summary_path


def evaluate_checkpoint(
    config_path: str | Path,
    checkpoint_path: str | Path,
    split_name: str = "test",
    output_dir: str | Path = "outputs/evaluate",
) -> Path:
    """Load a frozen checkpoint and evaluate it without retraining."""
    if split_name not in {"validation", "test"}:
        raise ValueError("split_name must be validation or test")
    config_source = Path(config_path).resolve()
    config = load_config(config_source)
    splits = build_splits(config)
    checkpoint = Path(checkpoint_path)
    policy = MaskedMLPPolicy.load(checkpoint)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "resolved_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    split_manifest = dataset_manifest(splits[split_name], split_name)
    evaluation_manifests = {split_name: split_manifest}
    (destination / "dataset_manifest.json").write_text(
        json.dumps(evaluation_manifests, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    random_seed = int(config["evaluation"].get("random_seed", 991))
    failure_penalty_ratio = config["evaluation"]["failure_penalty_ratio"]
    metrics, _ = evaluate_split(
        splits[split_name],
        policy,
        split_name,
        destination,
        random_seed,
        config["evaluation"].get("schedulers"),
        config_source.parent,
        failure_penalty_ratio,
    )
    payload = {
        "format_version": 2,
        "mode": "checkpoint_evaluation",
        "split": split_name,
        "checkpoint": {
            "path": checkpoint.name,
            "sha256": _file_sha256(checkpoint),
        },
        "dataset": split_manifest,
        "metrics": metrics,
        "scoring": {
            "failure_penalty_ratio": failure_penalty_ratio,
            "failures_remain_in_denominator": True,
            "publishable_requires_zero_failures": True,
        },
        "run_manifest": "run_manifest.json",
    }
    summary_path = destination / "evaluation_summary.json"
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_run_manifest(
        destination,
        mode="checkpoint_evaluation",
        split=split_name,
        config_source=config_source,
        config=config,
        dataset_manifests=evaluation_manifests,
        checkpoint=checkpoint,
        artifact_names=[
            "resolved_config.json",
            "dataset_manifest.json",
            f"{split_name}_per_instance.csv",
            f"{split_name}_failures.jsonl",
            f"{split_name}_example_schedule.json",
            "evaluation_summary.json",
        ],
    )
    ratio = metrics[policy.name]["mean_ratio"]
    print(
        f"done: {split_name} mean_ratio={ratio:.4f} " "(loaded checkpoint; no training)"
    )
    print(f"summary: {summary_path.resolve()}")
    return summary_path


def generate_scenarios(config_path: str | Path, destination: str | Path) -> None:
    config = load_config(config_path)
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    splits = build_splits(config)
    for split, scenarios in splits.items():
        split_dir = output / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for scenario in scenarios:
            scenario.save(split_dir / f"{scenario.id}.json")
    print(f"generated scenarios under {output.resolve()}")


def validate_scenario_file(path: str | Path) -> bool:
    """Print a stable JSON diagnostic and return whether a Scenario is valid."""

    try:
        scenario = Scenario.load(path)
    except ScenarioValidationError as error:
        print(
            json.dumps(
                {"valid": False, "error": error.to_dict()},
                ensure_ascii=False,
            )
        )
        return False
    print(
        json.dumps(
            {
                "valid": True,
                "scenario_id": scenario.id,
                "task_count": scenario.task_count,
                "resource_count": scenario.resource_count,
                "content_hash": scenario.content_hash(),
            },
            ensure_ascii=False,
        )
    )
    return True


def _evaluation_failure_report(summary_path: Path) -> dict[str, Any] | None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("mode") == "checkpoint_evaluation":
        groups = [(str(summary["split"]), summary["metrics"])]
    else:
        groups = [
            ("validation", summary.get("validation", {})),
            ("test", summary.get("test", {})),
        ]
    failures = []
    total = 0
    for split_name, metrics in groups:
        for scheduler, values in metrics.items():
            failure_count = int(values.get("failure_count", 0))
            if not failure_count:
                continue
            total += failure_count
            failures.append(
                {
                    "split": split_name,
                    "scheduler": scheduler,
                    "failure_count": failure_count,
                    "failure_rate": values["failure_rate"],
                    "error_counts": values["error_counts"],
                }
            )
    if not failures:
        return None
    return {
        "code": "evaluation_contains_failures",
        "message": "evaluation completed with scheduler failures",
        "details": {
            "failure_count": total,
            "results": failures,
            "summary": summary_path.name,
            "run_manifest": "run_manifest.json",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trisched",
        description="Minimum executable cloud-edge-device DAG scheduling framework",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    pipeline = subparsers.add_parser(
        "pipeline", help="train, evaluate, and write summary.json"
    )
    pipeline.add_argument("--config", default="configs/smoke.json")
    pipeline.add_argument("--output", default=None)
    train_bc = subparsers.add_parser(
        "train-bc",
        help="freeze HEFT teachers and select a public-STG BC checkpoint",
    )
    train_bc.add_argument("--config", default="configs/stg_bc.json")
    train_bc.add_argument("--output", default=None)
    train_ppo = subparsers.add_parser(
        "train-ppo",
        help="train multi-seed masked PPO on public-STG train/validation",
    )
    train_ppo.add_argument("--config", default="configs/stg_ppo.json")
    train_ppo.add_argument("--output", default=None)
    train_ppo.add_argument(
        "--resume",
        action="store_true",
        help="resume each seed from its last complete PPO epoch state",
    )
    train_task_gnn = subparsers.add_parser(
        "train-task-gnn",
        help="train multi-seed task-GNN BC plus PPO on STG train/validation",
    )
    train_task_gnn.add_argument(
        "--config",
        default="configs/stg_task_gnn.json",
    )
    train_task_gnn.add_argument("--output", default=None)
    train_task_gnn.add_argument(
        "--resume",
        action="store_true",
        help="resume task-GNN seeds from their last complete PPO epoch state",
    )
    generate = subparsers.add_parser(
        "generate", help="materialize deterministic scenario JSON files"
    )
    generate.add_argument("--config", default="configs/smoke.json")
    generate.add_argument("--output", default="outputs/generated")
    evaluate = subparsers.add_parser(
        "evaluate", help="load a checkpoint and evaluate it without retraining"
    )
    evaluate.add_argument("--config", default="configs/smoke.json")
    evaluate.add_argument("--checkpoint", default="outputs/smoke/masked_mlp.npz")
    evaluate.add_argument("--split", choices=("validation", "test"), default="test")
    evaluate.add_argument("--output", default="outputs/evaluate")
    validate = subparsers.add_parser(
        "validate-scenario", help="validate one Scenario JSON with structured errors"
    )
    validate.add_argument("--input", required=True)
    claim_gate = subparsers.add_parser(
        "claim-test-gate",
        help="atomically claim the frozen one-time public-test gate",
    )
    claim_gate.add_argument(
        "--contract",
        default="configs/p1_b02_evaluation_contract.json",
    )
    claim_gate.add_argument("--authorization", required=True)
    claim_gate.add_argument("--receipt", required=True)
    build_report = subparsers.add_parser(
        "build-report",
        help="validate a frozen multi-seed evidence package and aggregate it",
    )
    build_report.add_argument(
        "--contract",
        default="configs/p1_b02_evaluation_contract.json",
    )
    build_report.add_argument("--evidence", required=True)
    build_report.add_argument("--output", required=True)
    build_report.add_argument("--test-receipt", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "pipeline":
            summary_path = run_pipeline(args.config, args.output)
            failure = _evaluation_failure_report(summary_path)
            if failure is not None:
                print(
                    json.dumps({"ok": False, "error": failure}, ensure_ascii=False),
                    file=sys.stderr,
                )
                return 3
        elif args.command == "train-bc":
            run_bc_pipeline(args.config, args.output)
        elif args.command == "train-ppo":
            run_ppo_pipeline(args.config, args.output, resume=args.resume)
        elif args.command == "train-task-gnn":
            run_task_gnn_pipeline(args.config, args.output, resume=args.resume)
        elif args.command == "generate":
            generate_scenarios(args.config, args.output)
        elif args.command == "evaluate":
            summary_path = evaluate_checkpoint(
                args.config, args.checkpoint, args.split, args.output
            )
            failure = _evaluation_failure_report(summary_path)
            if failure is not None:
                print(
                    json.dumps({"ok": False, "error": failure}, ensure_ascii=False),
                    file=sys.stderr,
                )
                return 3
        elif args.command == "validate-scenario":
            if not validate_scenario_file(args.input):
                return 2
        elif args.command == "claim-test-gate":
            receipt_path = claim_public_test_gate(
                args.contract,
                args.authorization,
                args.receipt,
            )
            print(receipt_path.resolve())
        elif args.command == "build-report":
            report_path = build_evaluation_report(
                args.contract,
                args.evidence,
                args.output,
                test_receipt_path=args.test_receipt,
            )
            print(report_path.resolve())
        else:
            raise AssertionError(f"unknown command: {args.command}")
    except EvaluationReportError as error:
        print(
            json.dumps({"ok": False, "error": error.to_dict()}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 4
    except BehaviorCloningError as error:
        print(
            json.dumps({"ok": False, "error": error.to_dict()}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2
    except SchedulerAdapterError as error:
        print(
            json.dumps({"ok": False, "error": error.to_dict()}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
