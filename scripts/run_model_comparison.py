from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from scripts.compare_task_gnn_mlp import compare
from trisched.ppo import (
    load_ppo_config,
    load_task_gnn_config,
    run_ppo_pipeline,
    run_task_gnn_pipeline,
)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read comparison config {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("comparison config must be a JSON object")
    return value


def _resolve(source: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = source.parent / path
    return path.resolve()


def _positive_integer(value: Any, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be a {qualifier} integer")
    return value


def load_model_comparison_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = _load_object(source)
    allowed = {
        "format_version",
        "output_dir",
        "masked_mlp_config",
        "task_gnn_config",
        "comparison",
    }
    unknown = sorted(set(payload) - allowed)
    missing = sorted(allowed - set(payload))
    if unknown or missing:
        raise ValueError(
            f"comparison config keys are invalid; missing={missing}, unknown={unknown}"
        )
    if payload.get("format_version") != 1:
        raise ValueError("comparison config format_version must be 1")
    output_dir = _resolve(source, payload["output_dir"], "output_dir")
    mlp_config_path = _resolve(
        source,
        payload["masked_mlp_config"],
        "masked_mlp_config",
    )
    task_gnn_config_path = _resolve(
        source,
        payload["task_gnn_config"],
        "task_gnn_config",
    )
    comparison = payload.get("comparison")
    if not isinstance(comparison, dict):
        raise ValueError("comparison must be an object")
    comparison_allowed = {
        "bootstrap_samples",
        "bootstrap_seed",
        "latency_repeats",
    }
    comparison_unknown = sorted(set(comparison) - comparison_allowed)
    if comparison_unknown:
        raise ValueError(f"comparison contains unknown keys: {comparison_unknown}")
    bootstrap_seed = comparison.get("bootstrap_seed", 20260717)
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise ValueError("comparison.bootstrap_seed must be an integer")
    return {
        "format_version": 1,
        "config_source": source,
        "output_dir": output_dir,
        "masked_mlp_config": mlp_config_path,
        "task_gnn_config": task_gnn_config_path,
        "comparison": {
            "bootstrap_samples": _positive_integer(
                comparison.get("bootstrap_samples", 10_000),
                "comparison.bootstrap_samples",
            ),
            "bootstrap_seed": bootstrap_seed,
            "latency_repeats": _positive_integer(
                comparison.get("latency_repeats", 1),
                "comparison.latency_repeats",
                allow_zero=True,
            ),
        },
    }


def _resolved_data_path(config_source: Path, value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = config_source.parent / candidate
    return candidate.resolve()


def _validate_model_contracts(
    mlp_config_path: Path,
    task_gnn_config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    mlp = load_ppo_config(mlp_config_path)
    task_gnn = load_task_gnn_config(task_gnn_config_path)
    comparisons = {
        "seeds": (mlp["seeds"], task_gnn["seeds"]),
        "selected features": (
            mlp["features"]["selected"],
            task_gnn["features"]["selected"],
        ),
        "behavior cloning": (
            mlp["behavior_cloning"],
            task_gnn["behavior_cloning"],
        ),
        "PPO": (mlp["ppo"], task_gnn["ppo"]),
        "selection": (mlp["selection"], task_gnn["selection"]),
    }
    mismatched = [name for name, (left, right) in comparisons.items() if left != right]
    if mismatched:
        raise ValueError(
            "Masked MLP and Task-GNN must use the same comparison contract; "
            f"mismatched: {mismatched}"
        )
    mlp_manifest = _resolved_data_path(
        mlp_config_path,
        mlp["benchmark"]["manifest"],
    )
    gnn_manifest = _resolved_data_path(
        task_gnn_config_path,
        task_gnn["benchmark"]["manifest"],
    )
    mlp_raw_root = _resolved_data_path(
        mlp_config_path,
        mlp["benchmark"]["raw_root"],
    )
    gnn_raw_root = _resolved_data_path(
        task_gnn_config_path,
        task_gnn["benchmark"]["raw_root"],
    )
    if mlp_manifest != gnn_manifest or mlp_raw_root != gnn_raw_root:
        raise ValueError("Masked MLP and Task-GNN must use the same benchmark bytes")
    return mlp, task_gnn, mlp_manifest, mlp_raw_root


def _resolved_run_payload(
    config: dict[str, Any],
    mlp: dict[str, Any],
    task_gnn: dict[str, Any],
    manifest_path: Path,
    raw_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "mode": "masked_mlp_task_gnn_comparison",
        "config_source": {
            "name": config["config_source"].name,
            "sha256": _file_hash(config["config_source"]),
        },
        "output_dir": str(output_dir),
        "model_configs": {
            "masked_mlp": {
                "path": str(config["masked_mlp_config"]),
                "sha256": _file_hash(config["masked_mlp_config"]),
            },
            "task_gnn": {
                "path": str(config["task_gnn_config"]),
                "sha256": _file_hash(config["task_gnn_config"]),
            },
        },
        "shared_contract": {
            "seeds": mlp["seeds"],
            "benchmark_manifest": str(manifest_path),
            "benchmark_manifest_sha256": _file_hash(manifest_path),
            "raw_root": str(raw_root),
            "features": mlp["features"]["selected"],
            "behavior_cloning": mlp["behavior_cloning"],
            "ppo": mlp["ppo"],
            "selection": mlp["selection"],
            "test_accessed": False,
        },
        "task_gnn": task_gnn["task_gnn"],
        "comparison": config["comparison"],
        "output_layout": {
            "masked_mlp": "masked_mlp",
            "task_gnn": "task_gnn",
            "results": "results",
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _model_resume_requested(output: Path, resume: bool) -> bool:
    return bool(resume and output.is_dir() and any(output.iterdir()))


def run_model_comparison(
    config_path: str | Path,
    output_override: str | Path | None = None,
    *,
    resume: bool = False,
) -> Path:
    """Train both models, compare validation evidence, and render results."""

    config = load_model_comparison_config(config_path)
    output_dir = (
        config["output_dir"]
        if output_override is None
        else Path(output_override).resolve()
    )
    mlp, task_gnn, manifest_path, raw_root = _validate_model_contracts(
        config["masked_mlp_config"],
        config["task_gnn_config"],
    )
    resolved = _resolved_run_payload(
        config,
        mlp,
        task_gnn,
        manifest_path,
        raw_root,
        output_dir,
    )
    resolved_path = output_dir / "resolved_comparison_config.json"
    if resume:
        if not output_dir.is_dir() or not any(output_dir.iterdir()):
            raise ValueError(f"resume output does not exist or is empty: {output_dir}")
        if not resolved_path.is_file():
            raise ValueError("resume output is missing resolved_comparison_config.json")
        previous = _load_object(resolved_path)
        if previous != resolved:
            raise ValueError(
                "comparison config, model contract, or output path changed"
            )
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError(f"refusing to mix comparison runs in {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(resolved_path, resolved)

    mlp_output = output_dir / "masked_mlp"
    task_gnn_output = output_dir / "task_gnn"
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] training and validating Masked MLP")
    started = time.perf_counter()
    mlp_summary = run_ppo_pipeline(
        config["masked_mlp_config"],
        mlp_output,
        resume=_model_resume_requested(mlp_output, resume),
    )
    mlp_wall_seconds = time.perf_counter() - started

    print("[2/4] training and validating Task-GNN")
    started = time.perf_counter()
    task_gnn_summary = run_task_gnn_pipeline(
        config["task_gnn_config"],
        task_gnn_output,
        resume=_model_resume_requested(task_gnn_output, resume),
    )
    task_gnn_wall_seconds = time.perf_counter() - started

    print("[3/4] building paired validation comparison")
    comparison_path = results_dir / "comparison.json"
    comparison_args = argparse.Namespace(
        mlp_root=str(mlp_output),
        task_gnn_root=str(task_gnn_output),
        output=str(comparison_path),
        benchmark_manifest=str(manifest_path),
        raw_root=str(raw_root),
        bootstrap_samples=config["comparison"]["bootstrap_samples"],
        bootstrap_seed=config["comparison"]["bootstrap_seed"],
        latency_repeats=config["comparison"]["latency_repeats"],
        mlp_wall_seconds=mlp_wall_seconds,
        task_gnn_wall_seconds=task_gnn_wall_seconds,
    )
    compare(comparison_args)

    print("[4/4] writing top-level summary and manifest")
    comparison = _load_object(comparison_path)
    pipeline_summary = {
        "format_version": 1,
        "mode": "masked_mlp_task_gnn_train_validate_compare",
        "test_accessed": False,
        "models": {
            "masked_mlp": {
                "summary": mlp_summary.relative_to(output_dir).as_posix(),
                "wall_seconds": mlp_wall_seconds,
                "mean_ratio": comparison["paired_validation"]["masked_mlp_mean_ratio"],
            },
            "task_gnn": {
                "summary": task_gnn_summary.relative_to(output_dir).as_posix(),
                "wall_seconds": task_gnn_wall_seconds,
                "mean_ratio": comparison["paired_validation"]["task_gnn_mean_ratio"],
            },
        },
        "paired_validation": comparison["paired_validation"],
        "development_gate_passed": comparison["development_gate_passed"],
        "recommendation": comparison["recommendation"],
        "results": {
            "json": "results/comparison.json",
            "html": "results/comparison.html",
            "svg": "results/comparison.svg",
            "per_instance_csv": "results/comparison_per_instance.csv",
            "per_seed_csv": "results/comparison_per_seed.csv",
            "per_scenario_csv": "results/comparison_per_scenario.csv",
            "manifest": "results/comparison_manifest.json",
        },
    }
    summary_path = output_dir / "comparison_pipeline_summary.json"
    _write_json(summary_path, pipeline_summary)
    manifest_artifacts = (
        resolved_path,
        summary_path,
        mlp_summary,
        mlp_output / "ppo_run_manifest.json",
        task_gnn_summary,
        task_gnn_output / "task_gnn_run_manifest.json",
        comparison_path,
        results_dir / "comparison.html",
        results_dir / "comparison.svg",
        results_dir / "comparison_per_instance.csv",
        results_dir / "comparison_per_seed.csv",
        results_dir / "comparison_per_scenario.csv",
        results_dir / "comparison_manifest.json",
    )
    pipeline_manifest = {
        "format_version": 1,
        "mode": "model_comparison_pipeline_manifest",
        "hash_algorithm": "sha256",
        "artifacts": {
            path.relative_to(output_dir).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": _file_hash(path),
            }
            for path in manifest_artifacts
        },
    }
    _write_json(output_dir / "comparison_pipeline_manifest.json", pipeline_manifest)
    print(f"done: {summary_path.resolve()}")
    print(f"visual report: {(results_dir / 'comparison.html').resolve()}")
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train Masked MLP and Task-GNN, validate both, compare paired "
            "performance, and render result files"
        )
    )
    parser.add_argument(
        "--config",
        default=REPOSITORY / "configs" / "stg_model_comparison.json",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume either model from its last complete PPO epoch",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_model_comparison(args.config, args.output, resume=args.resume)
    except Exception as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
