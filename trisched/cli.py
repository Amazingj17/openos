from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .evaluation import dataset_manifest, evaluate_split, write_summary
from .learning import MaskedMLPPolicy, train_policy
from .scenario import Scenario, generate_dataset


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
    config = load_config(config_path)
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
        name: dataset_manifest(scenarios, name)
        for name, scenarios in splits.items()
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

    print("[3/4] evaluating HEFT, Greedy-EFT, Random and Masked-MLP")
    random_seed = int(config["evaluation"].get("random_seed", 991))
    validation_metrics, _ = evaluate_split(
        splits["validation"],
        policy,
        "validation",
        output_dir,
        random_seed,
    )
    test_metrics, _ = evaluate_split(
        splits["test"], policy, "test", output_dir, random_seed
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
    ratio = test_metrics["masked_mlp"]["mean_ratio"]
    print(f"done: test mean_ratio={ratio:.4f} (lower is better; HEFT=1.0)")
    print(f"summary: {summary_path.resolve()}")
    return summary_path


def generate_scenarios(
    config_path: str | Path, destination: str | Path
) -> None:
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
    generate = subparsers.add_parser(
        "generate", help="materialize deterministic scenario JSON files"
    )
    generate.add_argument("--config", default="configs/smoke.json")
    generate.add_argument("--output", default="outputs/generated")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "pipeline":
        run_pipeline(args.config, args.output)
    elif args.command == "generate":
        generate_scenarios(args.config, args.output)
    else:
        raise AssertionError(f"unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

