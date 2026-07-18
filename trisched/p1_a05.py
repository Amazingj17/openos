from __future__ import annotations

import json
import os
import re
import shutil
import sys
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from . import __version__
from .bc import (
    BehaviorCloningError,
    _git_metadata,
    _write_json,
    _write_jsonl,
    build_teacher_manifest,
    evaluate_bc_policy,
    freeze_teacher_dataset,
    policy_parameter_hash,
)
from .benchmark import load_benchmark_manifest, load_frozen_split
from .hashing import (
    canonical_json_sha256,
    file_sha256,
    portable_text_hashes,
)
from .learning import FEATURE_NAMES, TEACHER_FEATURE_NAMES, MaskedMLPPolicy
from .ood import load_materialized_development_slices
from .ppo import (
    ValueNetwork,
    _checkpoint_metadata,
    _publish_resume_staging,
    train_masked_ppo,
)
from .scenario import Scenario, generate_scenario


PREPARED_MANIFEST_NAME = "p1_a05_training_input_manifest.json"
DRY_RUN_NAME = "p1_a05_rollout_dry_run.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40,64}")
_FORBIDDEN_DATA_NAME = re.compile(
    r"public[-_]?test|claim[-_]?test|test[-_]?archive",
    re.IGNORECASE,
)
_REVIEW_GATES = (
    "synthetic_manifest_and_content_disjoint",
    "rollout_dry_run_90_episodes_6000_transitions",
    "strict_config_rejector",
    "public_test_bytes_absent",
    "warm_start_bytes_bound",
    "full_regression_passed",
)


def _fail(
    code: str,
    path: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> None:
    raise BehaviorCloningError(code, path, message, details=details)


def _load_json(path: Path, error_path: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail("p1_a05_read", error_path, str(error))
    if not isinstance(value, dict):
        _fail("p1_a05_type", error_path, "expected an object")
    return value


def _strict_keys(
    value: Any,
    *,
    required: set[str],
    path: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail("p1_a05_config_type", path, "expected an object")
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required)
    if missing or unknown:
        _fail(
            "p1_a05_config_variable",
            path,
            "P1-A05 rejects missing or unreviewed variables",
            details={"missing": missing, "unknown": unknown},
        )
    return value


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("p1_a05_config_value", path, "expected a non-empty string")
    return value


def _resolve(config_source: Path, value: str) -> Path:
    path = Path(value)
    return (
        path.resolve()
        if path.is_absolute()
        else (config_source.parent / path).resolve()
    )


def _repository() -> Path:
    return Path(__file__).resolve().parents[1]


def _expected_ppo(base: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base["ppo"])
    result["episodes_per_epoch"] = 90
    return result


def load_p1_a05_config(path: str | Path) -> dict[str, Any]:
    """Load the single frozen rollout-source intervention and reject all extras."""

    source = Path(path).resolve()
    payload = _load_json(source, "$")
    required_top = {
        "format_version",
        "task_id",
        "preregister",
        "output_dir",
        "prepared_input_dir",
        "implementation_review",
        "benchmark",
        "development",
        "warm_start",
        "model",
        "seeds",
        "features",
        "ppo",
        "selection",
        "synthetic_generator",
        "rollout_plan",
        "public_test",
    }
    _strict_keys(payload, required=required_top, path="$")
    if payload["format_version"] != 1:
        _fail("p1_a05_config_version", "$.format_version", "expected 1")
    if payload["task_id"] != "P1-A05-SIZE-ROBUSTNESS":
        _fail("p1_a05_config_task", "$.task_id", "wrong task id")

    prereg_binding = _strict_keys(
        payload["preregister"],
        required={"path", "canonical_sha256"},
        path="$.preregister",
    )
    prereg_path = _resolve(
        source,
        _nonempty_string(prereg_binding["path"], "$.preregister.path"),
    )
    preregister = _load_json(prereg_path, "$.preregister.path")
    preregister_hash = canonical_json_sha256(preregister)
    if (
        prereg_binding["canonical_sha256"] != preregister_hash
        or preregister_hash
        != "9ab60842d87e6d906ab18ad135cda2a372be569ad39b28ce6f17c60f1ff64647"
    ):
        _fail(
            "p1_a05_preregister_hash",
            "$.preregister.canonical_sha256",
            "preregister binding changed",
        )

    base_binding = preregister["evidence_bindings"]["base_training_config"]
    base_path = _repository() / base_binding["path"]
    base = _load_json(base_path, "$.preregister.base_training_config")
    if canonical_json_sha256(base) != base_binding["canonical_sha256"]:
        _fail(
            "p1_a05_base_hash",
            "$.preregister.base_training_config",
            "base training config changed",
        )

    for key in ("output_dir", "prepared_input_dir", "implementation_review"):
        _nonempty_string(payload[key], f"$.{key}")
    benchmark = _strict_keys(
        payload["benchmark"],
        required={"manifest", "raw_root"},
        path="$.benchmark",
    )
    if benchmark != base["benchmark"]:
        _fail(
            "p1_a05_config_variable",
            "$.benchmark",
            "benchmark binding must match P1-A04",
        )
    development = _strict_keys(
        payload["development"],
        required={"contract", "materialization_root", "manifest"},
        path="$.development",
    )
    for key, value in development.items():
        _nonempty_string(value, f"$.development.{key}")
    contract = _load_json(
        _resolve(source, development["contract"]), "$.development.contract"
    )
    expected_contract_hash = preregister["evidence_bindings"]["evaluation_contract"][
        "canonical_sha256"
    ]
    if canonical_json_sha256(contract) != expected_contract_hash:
        _fail(
            "p1_a05_contract_hash",
            "$.development.contract",
            "evaluation contract changed",
        )

    model = _strict_keys(
        payload["model"],
        required={"policy_class", "hidden_dim"},
        path="$.model",
    )
    frozen = preregister["frozen_invariants"]
    if model != {
        "policy_class": frozen["policy_class"],
        "hidden_dim": frozen["hidden_dim"],
    }:
        _fail("p1_a05_model", "$.model", "frozen model changed")
    if payload["seeds"] != base["seeds"]:
        _fail("p1_a05_seeds", "$.seeds", "training seeds changed")
    features = _strict_keys(
        payload["features"], required={"exclude"}, path="$.features"
    )
    if features != base["features"]:
        _fail("p1_a05_features", "$.features", "feature schema changed")
    ppo = _strict_keys(payload["ppo"], required=set(base["ppo"]), path="$.ppo")
    if ppo != _expected_ppo(base):
        changed = sorted(
            key for key in ppo if ppo.get(key) != _expected_ppo(base).get(key)
        )
        _fail(
            "p1_a05_config_variable",
            "$.ppo",
            "only the coupled episodes_per_epoch change is allowed",
            details={"changed": changed},
        )
    selection = _strict_keys(
        payload["selection"], required=set(base["selection"]), path="$.selection"
    )
    if selection != base["selection"]:
        _fail("p1_a05_selection", "$.selection", "selection rule changed")

    expected_generator = {
        key: preregister["single_intervention"]["synthetic_generator"][key]
        for key in (
            "callable",
            "scenario_id_template",
            "task_count",
            "resource_count",
            "edge_probability",
        )
    }
    generator = _strict_keys(
        payload["synthetic_generator"],
        required=set(expected_generator),
        path="$.synthetic_generator",
    )
    if generator != expected_generator:
        _fail(
            "p1_a05_generator",
            "$.synthetic_generator",
            "synthetic generator changed",
        )
    if (
        payload["rollout_plan"]
        != preregister["single_intervention"]["ppo_rollout_plan"]
    ):
        _fail("p1_a05_rollout_plan", "$.rollout_plan", "rollout plan changed")
    if payload["public_test"] != "forbidden":
        _fail("p1_a05_public_test", "$.public_test", "public test is forbidden")

    warm = _strict_keys(
        payload["warm_start"],
        required={"directory", "summary_sha256", "checkpoints"},
        path="$.warm_start",
    )
    _nonempty_string(warm["directory"], "$.warm_start.directory")
    if (
        warm["summary_sha256"]
        != preregister["evidence_bindings"]["base_training_summary"]["sha256"]
    ):
        _fail(
            "p1_a05_warm_start",
            "$.warm_start.summary_sha256",
            "P1-A04 summary binding changed",
        )
    if not isinstance(warm["checkpoints"], list):
        _fail("p1_a05_config_type", "$.warm_start.checkpoints", "expected an array")
    expected_seeds = list(base["seeds"])
    observed_seeds: list[int] = []
    names: set[str] = set()
    for index, raw in enumerate(warm["checkpoints"]):
        item = _strict_keys(
            raw,
            required={"seed", "name", "sha256", "parameter_sha256"},
            path=f"$.warm_start.checkpoints[{index}]",
        )
        if not isinstance(item["seed"], int) or isinstance(item["seed"], bool):
            _fail(
                "p1_a05_config_value",
                f"$.warm_start.checkpoints[{index}].seed",
                "expected an integer seed",
            )
        observed_seeds.append(item["seed"])
        name = _nonempty_string(item["name"], f"$.warm_start.checkpoints[{index}].name")
        if Path(name).name != name or name in names:
            _fail(
                "p1_a05_warm_start",
                f"$.warm_start.checkpoints[{index}].name",
                "checkpoint name must be a unique basename",
            )
        names.add(name)
        for field in ("sha256", "parameter_sha256"):
            if (
                not isinstance(item[field], str)
                or _SHA256.fullmatch(item[field]) is None
            ):
                _fail(
                    "p1_a05_config_value",
                    f"$.warm_start.checkpoints[{index}].{field}",
                    "expected lowercase SHA-256",
                )
    if observed_seeds != expected_seeds:
        _fail(
            "p1_a05_warm_start",
            "$.warm_start.checkpoints",
            "warm-start seeds must match the frozen training seed order",
        )
    return payload


def _copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    if not source.is_file() or file_sha256(source) != expected_sha256:
        _fail("p1_a05_source_hash", "$source", f"source missing or changed: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if file_sha256(destination) != expected_sha256:
        _fail(
            "p1_a05_copy_hash", "$destination", f"copied bytes changed: {destination}"
        )


def _synthetic_seed_index_pairs(config: Mapping[str, Any]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for epoch in config["rollout_plan"]:
        spec = epoch["synthetic"]
        indices = range(spec["index_start_inclusive"], spec["index_end_inclusive"] + 1)
        seeds = range(spec["seed_start_inclusive"], spec["seed_end_inclusive"] + 1)
        pairs.extend(zip(indices, seeds))
    if pairs != list(zip(range(60), range(20261001, 20261061))):
        _fail("p1_a05_rollout_plan", "$.rollout_plan", "synthetic range changed")
    return pairs


def _forbidden_named_files(root: Path) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and _FORBIDDEN_DATA_NAME.search(path.name)
    )


def _load_prepared_synthetic(root: Path, manifest: Mapping[str, Any]) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for index, entry in enumerate(manifest["synthetic_scenarios"]):
        relative = PurePosixPath(entry["file"])
        if relative.is_absolute() or ".." in relative.parts:
            _fail(
                "p1_a05_prepared_input",
                f"$.synthetic_scenarios[{index}].file",
                "unsafe path",
            )
        path = root.joinpath(*relative.parts).resolve()
        if not path.is_relative_to(root) or file_sha256(path) != entry["file_sha256"]:
            _fail(
                "p1_a05_prepared_input",
                f"$.synthetic_scenarios[{index}].file_sha256",
                "synthetic scenario bytes changed",
            )
        scenario = Scenario.load(path)
        if (
            scenario.id != entry["scenario_id"]
            or scenario.seed != entry["seed"]
            or scenario.content_hash() != entry["scenario_hash"]
        ):
            _fail(
                "p1_a05_prepared_input",
                f"$.synthetic_scenarios[{index}]",
                "synthetic scenario metadata changed",
            )
        scenarios.append(scenario)
    return scenarios


def _prepared_paths(
    config_source: Path, config: Mapping[str, Any]
) -> tuple[Path, Path, Path]:
    root = _resolve(config_source, config["prepared_input_dir"])
    return root, root / PREPARED_MANIFEST_NAME, root / DRY_RUN_NAME


def _build_dry_run_report(
    config_source: Path,
    config: Mapping[str, Any],
    root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    benchmark_manifest = root / "benchmark_manifest.json"
    benchmark_raw = root / "benchmark_raw"
    train = load_frozen_split(
        benchmark_raw,
        benchmark_manifest,
        "train",
        purpose="teacher",
    )
    synthetic = _load_prepared_synthetic(root, manifest)
    epochs: list[dict[str, Any]] = []
    for plan in config["rollout_plan"]:
        stg_spec = plan["stg_train_split_indices"]
        synthetic_spec = plan["synthetic"]
        stg = train[stg_spec["start_inclusive"] : stg_spec["end_inclusive"] + 1]
        generated = synthetic[
            synthetic_spec["index_start_inclusive"] : synthetic_spec[
                "index_end_inclusive"
            ]
            + 1
        ]
        scenarios = [*stg, *generated]
        counts = Counter(scenario.task_count for scenario in scenarios)
        episode_count = len(scenarios)
        transition_count = sum(scenario.task_count for scenario in scenarios)
        if (
            episode_count != 90
            or transition_count != 6000
            or counts != {50: 60, 100: 30}
        ):
            _fail(
                "p1_a05_dry_run",
                f"$.rollout_plan[{plan['epoch'] - 1}]",
                "rollout budget differs from 60x50 + 30x100",
            )
        epochs.append(
            {
                "epoch": plan["epoch"],
                "scenario_ids": [scenario.id for scenario in scenarios],
                "episode_count": episode_count,
                "task_count_counts": {str(key): counts[key] for key in sorted(counts)},
                "transition_count": transition_count,
            }
        )
    return {
        "format_version": 1,
        "task": "P1-A05-ROLLOUT-DRY-RUN",
        "config_canonical_sha256": canonical_json_sha256(config),
        "prepared_manifest_sha256": file_sha256(root / PREPARED_MANIFEST_NAME),
        "epochs": epochs,
        "totals": {
            "epoch_count": 2,
            "episodes_per_seed": 180,
            "transitions_per_seed": 12000,
            "training_seed_count": 5,
            "formal_transition_count": 60000,
        },
        "checkpoint_loaded": False,
        "optimizer_created": False,
        "training_started": False,
        "public_test_accessed": False,
    }


def prepare_p1_a05_inputs(config_path: str | Path) -> Path:
    """Atomically build a train/validation-only root and the frozen synthetic set."""

    config_source = Path(config_path).resolve()
    config = load_p1_a05_config(config_source)
    target, _, _ = _prepared_paths(config_source, config)
    if target.exists():
        _fail(
            "p1_a05_output_exists",
            "$.prepared_input_dir",
            f"refusing to overwrite prepared input: {target}",
        )
    staging = target.with_name(f".{target.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir(parents=True)
    try:
        benchmark_source = _resolve(config_source, config["benchmark"]["manifest"])
        raw_source = _resolve(config_source, config["benchmark"]["raw_root"])
        benchmark = load_benchmark_manifest(benchmark_source)
        train = load_frozen_split(
            raw_source, benchmark_source, "train", purpose="teacher"
        )
        validation = load_frozen_split(
            raw_source,
            benchmark_source,
            "validation",
            purpose="model_selection",
        )
        benchmark_entries = [
            entry
            for entry in benchmark["entries"]
            if entry["split"] in {"train", "validation"}
        ]
        _copy_verified(
            benchmark_source,
            staging / "benchmark_manifest.json",
            file_sha256(benchmark_source),
        )
        copied_entries: list[dict[str, Any]] = []
        for entry in benchmark_entries:
            relative = PurePosixPath(entry["source"])
            source_path = raw_source.joinpath(*relative.parts)
            target_path = staging / "benchmark_raw" / Path(*relative.parts)
            _copy_verified(source_path, target_path, entry["source_sha256"])
            copied_entries.append(
                {
                    "split": entry["split"],
                    "split_index": entry["split_index"],
                    "source": entry["source"],
                    "source_sha256": entry["source_sha256"],
                    "scenario_id": entry["scenario_id"],
                    "scenario_hash": entry["scenario_hash"],
                    "task_count": entry["task_count"],
                }
            )

        development_root = _resolve(
            config_source, config["development"]["materialization_root"]
        )
        development_manifest_path = _resolve(
            config_source, config["development"]["manifest"]
        )
        development_slices, development_manifest = load_materialized_development_slices(
            _resolve(config_source, config["development"]["contract"]),
            development_root,
            development_manifest_path,
        )
        development_ids = {
            scenario.id
            for scenarios in development_slices.values()
            for scenario in scenarios
        }
        development_hashes = {
            scenario.content_hash()
            for scenarios in development_slices.values()
            for scenario in scenarios
        }
        benchmark_ids = {scenario.id for scenario in [*train, *validation]}
        benchmark_hashes = {
            scenario.content_hash() for scenario in [*train, *validation]
        }

        generator = config["synthetic_generator"]
        synthetic_entries: list[dict[str, Any]] = []
        synthetic_ids: set[str] = set()
        synthetic_hashes: set[str] = set()
        for index, seed in _synthetic_seed_index_pairs(config):
            scenario = generate_scenario(
                seed=seed,
                task_count=generator["task_count"],
                resource_count=generator["resource_count"],
                edge_probability=generator["edge_probability"],
                scenario_id=generator["scenario_id_template"].format(index=index),
            )
            scenario_hash = scenario.content_hash()
            if (
                scenario.id in benchmark_ids
                or scenario.id in development_ids
                or scenario.id in synthetic_ids
                or scenario_hash in benchmark_hashes
                or scenario_hash in development_hashes
                or scenario_hash in synthetic_hashes
            ):
                _fail(
                    "p1_a05_synthetic_overlap",
                    f"$.synthetic[{index}]",
                    "synthetic training scenario overlaps a frozen data set",
                )
            relative = PurePosixPath("synthetic", f"{index:04d}.json")
            scenario_path = staging.joinpath(*relative.parts)
            scenario_path.parent.mkdir(parents=True, exist_ok=True)
            scenario.save(scenario_path)
            synthetic_ids.add(scenario.id)
            synthetic_hashes.add(scenario_hash)
            synthetic_entries.append(
                {
                    "index": index,
                    "seed": seed,
                    "scenario_id": scenario.id,
                    "scenario_hash": scenario_hash,
                    "task_count": scenario.task_count,
                    "resource_count": scenario.resource_count,
                    "edge_count": len(scenario.edges),
                    "file": relative.as_posix(),
                    "file_sha256": file_sha256(scenario_path),
                }
            )

        warm_source = _resolve(config_source, config["warm_start"]["directory"])
        summary_path = warm_source / "ppo_summary.json"
        if file_sha256(summary_path) != config["warm_start"]["summary_sha256"]:
            _fail(
                "p1_a05_warm_start",
                "$.warm_start.summary_sha256",
                "P1-A04 summary bytes changed",
            )
        warm_entries: list[dict[str, Any]] = []
        for item in config["warm_start"]["checkpoints"]:
            source_path = warm_source / item["name"]
            target_path = staging / "warm_starts" / item["name"]
            _copy_verified(source_path, target_path, item["sha256"])
            actor = MaskedMLPPolicy.load(target_path)
            if (
                actor.seed != item["seed"]
                or actor.hidden_dim != config["model"]["hidden_dim"]
                or tuple(actor.feature_names)
                != tuple(
                    name for name in FEATURE_NAMES if name not in TEACHER_FEATURE_NAMES
                )
                or policy_parameter_hash(actor) != item["parameter_sha256"]
            ):
                _fail(
                    "p1_a05_warm_start",
                    f"$.warm_start.checkpoints.{item['seed']}",
                    "warm-start checkpoint metadata changed",
                )
            warm_entries.append(dict(item))

        test_sources = {
            entry["source"]
            for entry in benchmark["entries"]
            if entry["split"] == "test"
        }
        copied_test_count = sum(
            (staging / "benchmark_raw" / Path(*PurePosixPath(source).parts)).exists()
            for source in test_sources
        )
        if copied_test_count:
            _fail(
                "p1_a05_public_test",
                "$.prepared_input_dir",
                "public-test source bytes were copied",
            )
        manifest = {
            "format_version": 1,
            "task": "P1-A05-PREPARE-TRAINING-INPUT",
            "config": {
                "path": config_source.name,
                "canonical_sha256": canonical_json_sha256(config),
                "portable_text": portable_text_hashes(config_source),
            },
            "preregister_canonical_sha256": config["preregister"]["canonical_sha256"],
            "source_bindings": {
                "benchmark_manifest_sha256": file_sha256(benchmark_source),
                "development_manifest_sha256": file_sha256(development_manifest_path),
                "development_contract_canonical_sha256": canonical_json_sha256(
                    _load_json(
                        _resolve(config_source, config["development"]["contract"]),
                        "$.development.contract",
                    )
                ),
                "p1_a04_summary_sha256": file_sha256(summary_path),
            },
            "benchmark": {
                "benchmark_id": benchmark["benchmark_id"],
                "copied_splits": ["train", "validation"],
                "entries": copied_entries,
                "train_count": len(train),
                "validation_count": len(validation),
                "public_test_manifest_entry_count": len(test_sources),
                "public_test_file_count": copied_test_count,
            },
            "synthetic_scenarios": synthetic_entries,
            "warm_starts": warm_entries,
            "disjointness": {
                "synthetic_id_overlap_with_benchmark_train_validation": len(
                    synthetic_ids & benchmark_ids
                ),
                "synthetic_hash_overlap_with_benchmark_train_validation": len(
                    synthetic_hashes & benchmark_hashes
                ),
                "synthetic_id_overlap_with_development": len(
                    synthetic_ids & development_ids
                ),
                "synthetic_hash_overlap_with_development": len(
                    synthetic_hashes & development_hashes
                ),
            },
            "development": {
                "slice_count": len(development_slices),
                "scenario_count": sum(
                    len(items) for items in development_slices.values()
                ),
                "test_accessed": development_manifest["test_accessed"],
                "public_test_materialized": development_manifest[
                    "public_test_materialized"
                ],
            },
            "data_boundary": {
                "public_test_accessed": False,
                "public_test_files_copied": copied_test_count,
                "archive_file_count": len(list(staging.rglob("*.zip"))),
                "forbidden_named_file_count": len(_forbidden_named_files(staging)),
            },
            "training_started": False,
        }
        if any(manifest["disjointness"].values()) or any(
            manifest["data_boundary"].values()
        ):
            _fail(
                "p1_a05_prepared_input",
                "$.prepared_input_dir",
                "a training input gate is non-zero",
            )
        _write_json(staging / PREPARED_MANIFEST_NAME, manifest)
        dry_run = _build_dry_run_report(config_source, config, staging, manifest)
        _write_json(staging / DRY_RUN_NAME, dry_run)
        if _forbidden_named_files(staging):
            _fail(
                "p1_a05_public_test",
                "$.prepared_input_dir",
                "forbidden data name found in prepared root",
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, target)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return target / PREPARED_MANIFEST_NAME


def verify_p1_a05_prepared_inputs(
    config_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config_source = Path(config_path).resolve()
    config = load_p1_a05_config(config_source)
    root, manifest_path, dry_path = _prepared_paths(config_source, config)
    manifest = _load_json(manifest_path, "$.prepared_input.manifest")
    if (
        manifest.get("task") != "P1-A05-PREPARE-TRAINING-INPUT"
        or manifest.get("config", {}).get("canonical_sha256")
        != canonical_json_sha256(config)
        or manifest.get("preregister_canonical_sha256")
        != config["preregister"]["canonical_sha256"]
    ):
        _fail(
            "p1_a05_prepared_input",
            "$.prepared_input.manifest",
            "prepared manifest binding changed",
        )
    if manifest["benchmark"]["public_test_file_count"] != 0:
        _fail(
            "p1_a05_public_test",
            "$.prepared_input.manifest",
            "prepared root contains public-test files",
        )
    for entry in manifest["benchmark"]["entries"]:
        path = root / "benchmark_raw" / Path(*PurePosixPath(entry["source"]).parts)
        if file_sha256(path) != entry["source_sha256"]:
            _fail(
                "p1_a05_prepared_input",
                "$.prepared_input.benchmark",
                "copied benchmark bytes changed",
            )
    _load_prepared_synthetic(root, manifest)
    for item in manifest["warm_starts"]:
        path = root / "warm_starts" / item["name"]
        if file_sha256(path) != item["sha256"]:
            _fail(
                "p1_a05_prepared_input",
                "$.prepared_input.warm_starts",
                "warm-start bytes changed",
            )
    if _forbidden_named_files(root) or list(root.rglob("*.zip")):
        _fail(
            "p1_a05_public_test",
            "$.prepared_input",
            "prepared root contains a forbidden data file",
        )
    dry_run = _load_json(dry_path, "$.prepared_input.dry_run")
    expected_dry_run = _build_dry_run_report(config_source, config, root, manifest)
    if dry_run != expected_dry_run:
        _fail(
            "p1_a05_dry_run",
            "$.prepared_input.dry_run",
            "dry-run report changed",
        )
    return manifest, dry_run


def write_p1_a05_dry_run(config_path: str | Path) -> Path:
    """Reverify and rewrite only the no-policy rollout-count report."""

    config_source = Path(config_path).resolve()
    config = load_p1_a05_config(config_source)
    root, manifest_path, dry_path = _prepared_paths(config_source, config)
    manifest = _load_json(manifest_path, "$.prepared_input.manifest")
    report = _build_dry_run_report(config_source, config, root, manifest)
    _write_json(dry_path, report)
    return dry_path


def _load_implementation_review(
    config_source: Path,
    config: Mapping[str, Any],
    prepared_manifest_path: Path,
    dry_run_path: Path,
) -> tuple[Path, dict[str, Any]]:
    review_path = _resolve(config_source, config["implementation_review"])
    if not review_path.is_file():
        _fail(
            "p1_a05_review_missing",
            "$.implementation_review",
            "member B implementation review is required before checkpoint loading",
        )
    review = _load_json(review_path, "$.implementation_review")
    required = {
        "format_version",
        "task",
        "approved",
        "reviewer",
        "approved_source_commit",
        "config_canonical_sha256",
        "preregister_canonical_sha256",
        "prepared_manifest_sha256",
        "dry_run_sha256",
        "gates",
        "public_test",
    }
    _strict_keys(review, required=required, path="$.implementation_review")
    if (
        review["format_version"] != 1
        or review["task"] != "P1-A05-IMPLEMENT-REVIEW"
        or review["approved"] is not True
        or review["reviewer"] != "B"
        or not isinstance(review["approved_source_commit"], str)
        or _COMMIT.fullmatch(review["approved_source_commit"]) is None
        or review["config_canonical_sha256"] != canonical_json_sha256(config)
        or review["preregister_canonical_sha256"]
        != config["preregister"]["canonical_sha256"]
        or review["prepared_manifest_sha256"] != file_sha256(prepared_manifest_path)
        or review["dry_run_sha256"] != file_sha256(dry_run_path)
        or review["public_test"] != "forbidden"
    ):
        _fail(
            "p1_a05_review_invalid",
            "$.implementation_review",
            "implementation review binding is invalid",
        )
    if review["gates"] != {gate: True for gate in _REVIEW_GATES}:
        _fail(
            "p1_a05_review_invalid",
            "$.implementation_review.gates",
            "all frozen implementation gates must be true",
        )
    return review_path, review


def _combined_teacher_entries(
    benchmark: Mapping[str, Any],
    synthetic_entries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    train_entries = [
        dict(entry) for entry in benchmark["entries"] if entry["split"] == "train"
    ]
    for index, entry in enumerate(synthetic_entries):
        train_entries.append(
            {
                "split": "train",
                "split_index": 120 + index,
                "source": entry["file"],
                "source_sha256": entry["file_sha256"],
                "scenario_id": entry["scenario_id"],
                "scenario_hash": entry["scenario_hash"],
            }
        )
    return train_entries


def _formal_run_in_directory(
    config_source: Path,
    output_dir: Path,
    *,
    resume: bool,
) -> Path:
    config = load_p1_a05_config(config_source)
    prepared_root, prepared_manifest_path, dry_run_path = _prepared_paths(
        config_source, config
    )
    prepared_manifest, dry_run = verify_p1_a05_prepared_inputs(config_source)
    review_path, review = _load_implementation_review(
        config_source,
        config,
        prepared_manifest_path,
        dry_run_path,
    )
    repository = _repository()
    code_metadata = _git_metadata(repository)
    if code_metadata.get("working_tree_dirty") is not False:
        _fail(
            "p1_a05_formal_dirty",
            "$.code",
            "formal training requires a clean immutable worktree",
        )
    if code_metadata.get("commit") != review["approved_source_commit"]:
        _fail(
            "p1_a05_formal_commit",
            "$.code.commit",
            "current source commit is not the implementation commit approved by B",
            details={
                "expected": review["approved_source_commit"],
                "actual": code_metadata.get("commit"),
            },
        )
    if resume:
        previous = _load_json(
            output_dir / "resolved_config.json", "$.output.resolved_config"
        )
        if previous != config:
            _fail(
                "p1_a05_resume_config",
                "$.output.resolved_config",
                "resume config changed",
            )
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            _fail(
                "p1_a05_output_exists",
                "$.output_dir",
                "formal run output already exists",
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "resolved_config.json", config)

    benchmark_manifest_path = prepared_root / "benchmark_manifest.json"
    benchmark_raw = prepared_root / "benchmark_raw"
    benchmark = load_benchmark_manifest(benchmark_manifest_path)
    stg_train = load_frozen_split(
        benchmark_raw, benchmark_manifest_path, "train", purpose="teacher"
    )
    validation = load_frozen_split(
        benchmark_raw,
        benchmark_manifest_path,
        "validation",
        purpose="model_selection",
    )
    synthetic = _load_prepared_synthetic(prepared_root, prepared_manifest)
    all_train = [*stg_train, *synthetic]
    combined_entries = _combined_teacher_entries(
        benchmark, prepared_manifest["synthetic_scenarios"]
    )
    validation_entries = [
        entry for entry in benchmark["entries"] if entry["split"] == "validation"
    ]
    teacher_binding_hash = canonical_json_sha256(
        {
            "prepared_manifest_sha256": file_sha256(prepared_manifest_path),
            "combined_scenario_hashes": [
                scenario.content_hash() for scenario in all_train
            ],
        }
    )
    train_teacher = build_teacher_manifest(
        all_train,
        combined_entries,
        split="train",
        purpose="behavior_cloning_teacher",
        benchmark_manifest_name=PREPARED_MANIFEST_NAME,
        benchmark_manifest_sha256=teacher_binding_hash,
        benchmark_id="p1-a05-stg50-synthetic100-rollout-v1",
        code_metadata=code_metadata,
    )
    validation_reference = build_teacher_manifest(
        validation,
        validation_entries,
        split="validation",
        purpose="model_selection_reference",
        benchmark_manifest_name=benchmark_manifest_path.name,
        benchmark_manifest_sha256=file_sha256(benchmark_manifest_path),
        benchmark_id=benchmark["benchmark_id"],
        code_metadata=code_metadata,
    )
    teacher_artifacts = {
        "train_teacher_manifest.json": train_teacher,
        "validation_reference_manifest.json": validation_reference,
    }
    if resume:
        for name, expected in teacher_artifacts.items():
            if _load_json(output_dir / name, f"$.output.{name}") != expected:
                _fail(
                    "p1_a05_resume_artifact",
                    f"$.output.{name}",
                    "teacher artifact changed",
                )
    else:
        for name, value in teacher_artifacts.items():
            _write_json(output_dir / name, value)
        _write_jsonl(output_dir / "teacher_failures.jsonl", [])
    validation_states = freeze_teacher_dataset(
        validation,
        validation_reference,
        split="validation",
        purpose="model_selection_reference",
    )

    epoch_scenario_ids = [item["scenario_ids"] for item in dry_run["epochs"]]
    seed_results: list[dict[str, Any]] = []
    artifact_names = [
        "resolved_config.json",
        "train_teacher_manifest.json",
        "validation_reference_manifest.json",
        "teacher_failures.jsonl",
    ]
    resumed_seeds: list[int] = []
    ppo_config = {
        **config["ppo"],
        "failure_penalty_ratio": config["selection"]["failure_penalty_ratio"],
    }
    review_sha256 = file_sha256(review_path)
    for item in config["warm_start"]["checkpoints"]:
        seed = int(item["seed"])
        warm_source = prepared_root / "warm_starts" / item["name"]
        warm_output = output_dir / item["name"]
        if resume:
            if file_sha256(warm_output) != item["sha256"]:
                _fail(
                    "p1_a05_resume_artifact",
                    f"$.output.warm_start.{seed}",
                    "warm-start bytes changed",
                )
        else:
            _copy_verified(warm_source, warm_output, item["sha256"])
        warm_actor = MaskedMLPPolicy.load(warm_output)
        if policy_parameter_hash(warm_actor) != item["parameter_sha256"]:
            _fail(
                "p1_a05_warm_start",
                f"$.warm_start.{seed}",
                "warm-start parameter hash changed",
            )
        warm_metrics, _ = evaluate_bc_policy(
            warm_actor,
            validation,
            validation_reference,
            failure_penalty_ratio=config["selection"]["failure_penalty_ratio"],
            frozen_teacher_states=validation_states,
        )
        prefix = f"seed_{seed}"
        state_path = output_dir / f"{prefix}_ppo_training_state.npz"
        resume_seed = resume and state_path.is_file()
        if resume_seed:
            resumed_seeds.append(seed)
        resume_contract = {
            "format_version": 1,
            "task": "P1-A05-SIZE-ROBUSTNESS",
            "seed": seed,
            "config_canonical_sha256": canonical_json_sha256(config),
            "prepared_manifest_sha256": file_sha256(prepared_manifest_path),
            "dry_run_sha256": file_sha256(dry_run_path),
            "implementation_review_sha256": review_sha256,
            "approved_source_commit": review["approved_source_commit"],
            "code": code_metadata,
            "warm_start_parameter_sha256": item["parameter_sha256"],
            "rollout_epoch_scenario_ids": epoch_scenario_ids,
            "validation_scenario_hashes": [
                scenario.content_hash() for scenario in validation
            ],
            "public_test": "forbidden",
        }
        best, last, best_value, last_value, curve = train_masked_ppo(
            warm_actor,
            all_train,
            train_teacher,
            validation,
            validation_reference,
            ppo_config,
            seed=seed,
            validation_frozen_states=validation_states,
            resume_state_path=state_path,
            resume_contract=resume_contract,
            resume=resume_seed,
            epoch_scenario_ids=epoch_scenario_ids,
        )
        best_metrics, best_rows = evaluate_bc_policy(
            best,
            validation,
            validation_reference,
            failure_penalty_ratio=config["selection"]["failure_penalty_ratio"],
            frozen_teacher_states=validation_states,
        )
        last_metrics, last_rows = evaluate_bc_policy(
            last,
            validation,
            validation_reference,
            failure_penalty_ratio=config["selection"]["failure_penalty_ratio"],
            frozen_teacher_states=validation_states,
        )
        best_actor_path = output_dir / f"{prefix}_ppo_best_policy.npz"
        last_actor_path = output_dir / f"{prefix}_ppo_last_policy.npz"
        best_value_path = output_dir / f"{prefix}_ppo_best_value.npz"
        last_value_path = output_dir / f"{prefix}_ppo_last_value.npz"
        best.save(best_actor_path)
        last.save(last_actor_path)
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
                "behavior_cloning": "reused exact P1-A04 warm-start bytes",
                "ppo": curve,
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
        seed_results.append(
            {
                "seed": seed,
                "warm_start": {
                    **item,
                    "validation": warm_metrics,
                    "source": "exact P1-A04 bytes",
                },
                "selection": curve["selection"],
                "best_validation": best_metrics,
                "last_validation": last_metrics,
                "best_checkpoint": _checkpoint_metadata(
                    best_actor_path, best, best_value_path, best_value
                ),
                "last_checkpoint": _checkpoint_metadata(
                    last_actor_path, last, last_value_path, last_value
                ),
                "training_state": {
                    "name": state_path.name,
                    "sha256": file_sha256(state_path),
                    "completed_epoch": config["ppo"]["epochs"],
                    "boundary": "completed_ppo_epoch",
                },
                "improved_over_warm_start": best_metrics["mean_ratio"]
                < warm_metrics["mean_ratio"] - 1e-12,
                "selected_warm_start": curve["selection"]["best_epoch"] == 0,
            }
        )
        artifact_names.extend(
            [
                warm_output.name,
                best_actor_path.name,
                last_actor_path.name,
                best_value_path.name,
                last_value_path.name,
                state_path.name,
                curve_path.name,
                diagnostics_path.name,
                failures_path.name,
            ]
        )

    ratios = [float(item["best_validation"]["mean_ratio"]) for item in seed_results]
    validation_gate = all(
        item["best_validation"]["failure_count"] == 0
        and item["best_validation"]["illegal_action_count"] == 0
        and item["best_validation"]["mean_ratio"]
        <= config["selection"]["target_ratio"] + 1e-12
        for item in seed_results
    )
    summary = {
        "format_version": 1,
        "mode": "p1_a05_size_robustness",
        "task_id": config["task_id"],
        "data_access": {
            "prepared_input_manifest": PREPARED_MANIFEST_NAME,
            "loaded_splits": ["train", "validation"],
            "synthetic_training_scenarios": 60,
            "test_accessed": False,
            "public_test": "forbidden",
        },
        "single_intervention": "transition-budget-matched 50/100-task PPO rollout mixture",
        "rollout_dry_run": dry_run,
        "seeds": seed_results,
        "aggregate_validation": {
            "seed_count": len(seed_results),
            "mean_of_seed_mean_ratios": float(np.mean(ratios)),
            "std_of_seed_mean_ratios": float(np.std(ratios)),
            "failure_count": sum(
                int(item["best_validation"]["failure_count"]) for item in seed_results
            ),
            "illegal_action_count": sum(
                int(item["best_validation"]["illegal_action_count"])
                for item in seed_results
            ),
            "improved_seed_count": sum(
                int(item["improved_over_warm_start"]) for item in seed_results
            ),
            "warm_start_fallback_count": sum(
                int(item["selected_warm_start"]) for item in seed_results
            ),
        },
        "selection": {
            **config["selection"],
            "split": "validation",
            "test_accessed": False,
        },
        "validation_gate_passed": validation_gate,
        "implementation_review": {
            "path": review_path.name,
            "sha256": review_sha256,
            "approved_source_commit": review["approved_source_commit"],
        },
        "formal_run_count": 1,
        "run_manifest": "p1_a05_run_manifest.json",
    }
    summary_path = output_dir / "p1_a05_summary.json"
    _write_json(summary_path, summary)
    artifact_names.append(summary_path.name)
    artifacts = {
        name: {
            "bytes": (output_dir / name).stat().st_size,
            "sha256": file_sha256(output_dir / name),
        }
        for name in sorted(set(artifact_names))
    }
    source_hashes = {
        path.relative_to(repository).as_posix(): portable_text_hashes(path)
        for path in sorted((repository / "trisched").glob("*.py"))
    }
    run_manifest = {
        "format_version": 1,
        "mode": "p1_a05_size_robustness",
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "code": {**code_metadata, "portable_sources": source_hashes},
        "runtime": {
            "trisched": __version__,
            "python": sys.version,
            "numpy": np.__version__,
        },
        "execution": {
            "resume_requested": resume,
            "resumed_seeds": resumed_seeds,
            "publication_mode": "staging_directory_swap"
            if resume
            else "direct_new_directory",
            "formal_run_count": 1,
        },
        "inputs": {
            "config": {
                "name": config_source.name,
                "canonical_sha256": canonical_json_sha256(config),
                "portable_text": portable_text_hashes(config_source),
            },
            "preregister_canonical_sha256": config["preregister"]["canonical_sha256"],
            "prepared_manifest_sha256": file_sha256(prepared_manifest_path),
            "dry_run_sha256": file_sha256(dry_run_path),
            "implementation_review_sha256": review_sha256,
            "test_accessed": False,
            "public_test": "forbidden",
        },
        "checkpoints": {
            str(item["seed"]): {
                "best": item["best_checkpoint"],
                "last": item["last_checkpoint"],
                "training_state": item["training_state"],
            }
            for item in seed_results
        },
        "artifacts": artifacts,
    }
    _write_json(output_dir / "p1_a05_run_manifest.json", run_manifest)
    return summary_path


def run_p1_a05_pipeline(config_path: str | Path, *, resume: bool = False) -> Path:
    """Run the single formal candidate only after a bound B review receipt exists."""

    config_source = Path(config_path).resolve()
    config = load_p1_a05_config(config_source)
    prepared_root, manifest_path, dry_path = _prepared_paths(config_source, config)
    # Review validation intentionally happens before output creation and checkpoint loading.
    _load_implementation_review(config_source, config, manifest_path, dry_path)
    output_dir = _resolve(config_source, config["output_dir"])
    if not resume:
        return _formal_run_in_directory(config_source, output_dir, resume=False)
    if not output_dir.is_dir() or not any(output_dir.iterdir()):
        _fail(
            "p1_a05_resume_output",
            "$.output_dir",
            "resume output does not exist or is empty",
        )
    staging = output_dir.with_name(
        f".{output_dir.name}.p1-a05-resume-staging-{uuid.uuid4().hex}"
    )
    try:
        shutil.copytree(output_dir, staging)
        staged_summary = _formal_run_in_directory(config_source, staging, resume=True)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    _publish_resume_staging(staging, output_dir)
    return output_dir / staged_summary.name
