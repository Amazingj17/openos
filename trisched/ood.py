from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .benchmark import (
    BenchmarkValidationError,
    load_benchmark_manifest,
    load_frozen_split,
)
from .env import ScheduleResult, validate_schedule
from .oracle import validate_schedule_independent
from .reporting import (
    canonical_json_sha256,
    load_evaluation_contract,
)
from .scenario import (
    Edge,
    Resource,
    Scenario,
    ScenarioValidationError,
    generate_scenario,
)
from .schedulers import SchedulerAdapterError, SchedulerRunner


MATERIALIZER_VERSION = 1
MATERIALIZATION_MANIFEST_NAME = "development_slices_manifest.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class OODWorkflowError(RuntimeError):
    """Stable machine-readable error for P1-B02 OOD/evidence workflows."""

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
        return {
            "code": self.code,
            "path": self.path,
            "message": self.message,
            "details": self.details,
        }


def _fail(
    code: str,
    path: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> None:
    raise OODWorkflowError(code, path, message, details=details)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        _fail("ood_output_exists", "$output", "refusing to overwrite evidence output")
    except OSError as error:
        _fail("ood_output", "$output", str(error))


def _load_json(path: Path, *, error_path: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        _fail("ood_json", error_path, str(error))
    if not isinstance(payload, dict):
        _fail("ood_json", error_path, "expected a JSON object")
    return payload


def _slice_map(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in contract["slices"]}


def _positive_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("ood_contract", path, "expected a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        _fail("ood_contract", path, "expected a positive finite number")
    return result


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("ood_contract", path, "expected an integer")
    return value


def _scenario_id(template: Any, index: int, path: str) -> str:
    if not isinstance(template, str) or not template:
        _fail("ood_contract", path, "expected a non-empty scenario id template")
    try:
        scenario_id = template.format(index=index)
    except (IndexError, KeyError, ValueError) as error:
        _fail("ood_contract", path, f"invalid scenario id template: {error}")
    if (
        not scenario_id
        or "/" in scenario_id
        or "\\" in scenario_id
        or scenario_id in {".", ".."}
    ):
        _fail("ood_contract", path, "template produced an unsafe scenario id")
    return scenario_id


@dataclass(frozen=True)
class _MaterializationSpec:
    id_slice: dict[str, Any]
    size_slice: dict[str, Any]
    ccr_slice: dict[str, Any]
    system_slice: dict[str, Any]


def _materialization_spec(contract: dict[str, Any]) -> _MaterializationSpec:
    expected_development = ["id_validation", "ood_size", "ood_ccr", "ood_system"]
    if contract["modes"]["development"] != expected_development:
        _fail(
            "ood_contract",
            "$.modes.development",
            "expected the frozen ID/OOD development slice order",
            details={"expected": expected_development},
        )
    slices = _slice_map(contract)
    try:
        id_slice = slices["id_validation"]
        size_slice = slices["ood_size"]
        ccr_slice = slices["ood_ccr"]
        system_slice = slices["ood_system"]
    except KeyError as error:
        _fail("ood_contract", "$.slices", f"missing slice {error.args[0]}")

    id_source = id_slice.get("source")
    if not isinstance(id_source, dict):
        _fail("ood_contract", "$.slices.id_validation.source", "expected an object")
    if id_source.get("split") != "validation":
        _fail(
            "ood_contract",
            "$.slices.id_validation.source.split",
            "materialization may load validation only",
        )
    if id_source.get("purpose") != "model_selection":
        _fail(
            "ood_contract",
            "$.slices.id_validation.source.purpose",
            "expected the model_selection anti-test capability",
        )

    size_source = size_slice.get("source")
    if not isinstance(size_source, dict):
        _fail("ood_contract", "$.slices.ood_size.source", "expected an object")
    if size_source.get("generator") != "trisched.scenario.generate_scenario":
        _fail(
            "ood_contract",
            "$.slices.ood_size.source.generator",
            "unexpected generator",
        )
    seeds = size_source.get("generator_seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) != size_slice["scenario_count"]
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        _fail(
            "ood_contract",
            "$.slices.ood_size.source.generator_seeds",
            "expected one unique integer seed per scenario",
        )
    _integer(size_source.get("task_count"), "$.slices.ood_size.source.task_count")
    _integer(
        size_source.get("resource_count"),
        "$.slices.ood_size.source.resource_count",
    )
    edge_probability = size_source.get("edge_probability")
    if (
        isinstance(edge_probability, bool)
        or not isinstance(edge_probability, (int, float))
        or not 0.0 <= float(edge_probability) <= 1.0
    ):
        _fail(
            "ood_contract",
            "$.slices.ood_size.source.edge_probability",
            "expected a probability between zero and one",
        )
    _scenario_id(
        size_source.get("scenario_id_template"),
        0,
        "$.slices.ood_size.source.scenario_id_template",
    )

    ccr_source = ccr_slice.get("source")
    if (
        not isinstance(ccr_source, dict)
        or ccr_source.get("base_slice") != "id_validation"
    ):
        _fail("ood_contract", "$.slices.ood_ccr.source", "expected ID base slice")
    _positive_number(
        ccr_source.get("edge_data_scale"),
        "$.slices.ood_ccr.source.edge_data_scale",
    )
    _positive_number(
        ccr_source.get("off_diagonal_bandwidth_scale"),
        "$.slices.ood_ccr.source.off_diagonal_bandwidth_scale",
    )
    _scenario_id(
        ccr_source.get("scenario_id_template"),
        0,
        "$.slices.ood_ccr.source.scenario_id_template",
    )

    system_source = system_slice.get("source")
    if (
        not isinstance(system_source, dict)
        or system_source.get("base_slice") != "id_validation"
    ):
        _fail("ood_contract", "$.slices.ood_system.source", "expected ID base slice")
    multipliers = system_source.get("resource_speed_multipliers")
    if not isinstance(multipliers, list) or not multipliers:
        _fail(
            "ood_contract",
            "$.slices.ood_system.source.resource_speed_multipliers",
            "expected a non-empty array",
        )
    for index, value in enumerate(multipliers):
        _positive_number(
            value,
            f"$.slices.ood_system.source.resource_speed_multipliers[{index}]",
        )
    _positive_number(
        system_source.get("off_diagonal_bandwidth_scale"),
        "$.slices.ood_system.source.off_diagonal_bandwidth_scale",
    )
    _positive_number(
        system_source.get("off_diagonal_latency_scale"),
        "$.slices.ood_system.source.off_diagonal_latency_scale",
    )
    _scenario_id(
        system_source.get("scenario_id_template"),
        0,
        "$.slices.ood_system.source.scenario_id_template",
    )
    if (
        id_slice["scenario_count"] != ccr_slice["scenario_count"]
        or id_slice["scenario_count"] != system_slice["scenario_count"]
    ):
        _fail(
            "ood_contract",
            "$.slices",
            "ID-derived OOD slices must preserve scenario count",
        )
    return _MaterializationSpec(id_slice, size_slice, ccr_slice, system_slice)


def transform_ood_ccr(
    base: Scenario,
    *,
    scenario_id: str,
    edge_data_scale: float,
    off_diagonal_bandwidth_scale: float,
) -> Scenario:
    """Apply the frozen communication-stress transform without changing the base."""

    return Scenario(
        id=scenario_id,
        seed=base.seed,
        tasks=base.tasks,
        resources=base.resources,
        edges=tuple(
            Edge(edge.source, edge.target, float(edge.data * edge_data_scale))
            for edge in base.edges
        ),
        bandwidth=tuple(
            tuple(
                value if row == column else float(value * off_diagonal_bandwidth_scale)
                for column, value in enumerate(values)
            )
            for row, values in enumerate(base.bandwidth)
        ),
        latency=base.latency,
    )


def transform_ood_system(
    base: Scenario,
    *,
    scenario_id: str,
    resource_speed_multipliers: Sequence[float],
    off_diagonal_bandwidth_scale: float,
    off_diagonal_latency_scale: float,
) -> Scenario:
    """Apply the frozen resource/link-stress transform without changing the base."""

    if len(resource_speed_multipliers) != base.resource_count:
        _fail(
            "ood_transform",
            "$.slices.ood_system.source.resource_speed_multipliers",
            "resource multiplier count differs from the base scenario",
            details={
                "scenario_id": base.id,
                "expected": base.resource_count,
                "actual": len(resource_speed_multipliers),
            },
        )
    resources = tuple(
        Resource(
            resource.id,
            resource.name,
            resource.kind,
            float(resource.speed * resource_speed_multipliers[resource.id]),
        )
        for resource in base.resources
    )
    return Scenario(
        id=scenario_id,
        seed=base.seed,
        tasks=base.tasks,
        resources=resources,
        edges=base.edges,
        bandwidth=tuple(
            tuple(
                value if row == column else float(value * off_diagonal_bandwidth_scale)
                for column, value in enumerate(values)
            )
            for row, values in enumerate(base.bandwidth)
        ),
        latency=tuple(
            tuple(
                value if row == column else float(value * off_diagonal_latency_scale)
                for column, value in enumerate(values)
            )
            for row, values in enumerate(base.latency)
        ),
    )


def _validate_source_binding(
    source_binding: Mapping[str, Any],
    scenarios: Sequence[Scenario],
) -> list[dict[str, Any]]:
    required = {
        "benchmark_id",
        "manifest_name",
        "manifest_sha256",
        "split",
        "purpose",
        "scenario_hashes_sha256",
        "entries",
    }
    if set(source_binding) != required:
        _fail(
            "ood_source",
            "$source_binding",
            "source binding keys differ from the frozen schema",
            details={
                "missing": sorted(required - set(source_binding)),
                "unknown": sorted(set(source_binding) - required),
            },
        )
    if (
        not isinstance(source_binding["benchmark_id"], str)
        or not source_binding["benchmark_id"]
    ):
        _fail("ood_source", "$source_binding.benchmark_id", "invalid benchmark id")
    manifest_name = source_binding["manifest_name"]
    if (
        not isinstance(manifest_name, str)
        or not manifest_name
        or "/" in manifest_name
        or "\\" in manifest_name
        or manifest_name in {".", ".."}
    ):
        _fail(
            "ood_source",
            "$source_binding.manifest_name",
            "expected a safe manifest basename",
        )
    if (
        source_binding["split"] != "validation"
        or source_binding["purpose"] != "model_selection"
    ):
        _fail(
            "ood_source",
            "$source_binding",
            "only validation/model_selection source access is allowed",
        )
    if (
        not isinstance(source_binding["manifest_sha256"], str)
        or _SHA256.fullmatch(source_binding["manifest_sha256"]) is None
    ):
        _fail("ood_source", "$source_binding.manifest_sha256", "invalid SHA-256")
    if (
        not isinstance(source_binding["scenario_hashes_sha256"], str)
        or _SHA256.fullmatch(source_binding["scenario_hashes_sha256"]) is None
    ):
        _fail(
            "ood_source",
            "$source_binding.scenario_hashes_sha256",
            "invalid SHA-256",
        )
    entries = source_binding["entries"]
    if not isinstance(entries, list) or len(entries) != len(scenarios):
        _fail(
            "ood_source",
            "$source_binding.entries",
            "expected one source entry per validation scenario",
        )
    normalized: list[dict[str, Any]] = []
    observed_hashes: list[str] = []
    for index, (raw_entry, scenario) in enumerate(zip(entries, scenarios, strict=True)):
        if not isinstance(raw_entry, dict):
            _fail("ood_source", f"$source_binding.entries[{index}]", "expected object")
        expected = {
            "source",
            "source_sha256",
            "scenario_id",
            "scenario_hash",
            "split_index",
        }
        if set(raw_entry) != expected:
            _fail(
                "ood_source",
                f"$source_binding.entries[{index}]",
                "source entry keys differ from the frozen schema",
            )
        if raw_entry["scenario_id"] != scenario.id:
            _fail(
                "ood_source",
                f"$source_binding.entries[{index}].scenario_id",
                "source scenario id mismatch",
            )
        scenario_hash = scenario.content_hash()
        if raw_entry["scenario_hash"] != scenario_hash:
            _fail(
                "ood_source",
                f"$source_binding.entries[{index}].scenario_hash",
                "source scenario hash mismatch",
            )
        if raw_entry["split_index"] != index:
            _fail(
                "ood_source",
                f"$source_binding.entries[{index}].split_index",
                "validation split index mismatch",
            )
        raw_source = raw_entry["source"]
        if (
            not isinstance(raw_source, str)
            or not raw_source
            or "\\" in raw_source
            or PurePosixPath(raw_source).is_absolute()
            or ".." in PurePosixPath(raw_source).parts
        ):
            _fail(
                "ood_source",
                f"$source_binding.entries[{index}].source",
                "expected a safe relative POSIX source path",
            )
        if (
            not isinstance(raw_entry["source_sha256"], str)
            or _SHA256.fullmatch(raw_entry["source_sha256"]) is None
        ):
            _fail(
                "ood_source",
                f"$source_binding.entries[{index}].source_sha256",
                "invalid source SHA-256",
            )
        observed_hashes.append(scenario_hash)
        normalized.append(dict(raw_entry))
    if source_binding["scenario_hashes_sha256"] != canonical_json_sha256(
        observed_hashes
    ):
        _fail(
            "ood_source",
            "$source_binding.scenario_hashes_sha256",
            "validation scenario aggregate hash mismatch",
        )
    return normalized


def _validate_materialized_provenance(
    manifest: Mapping[str, Any],
    spec: _MaterializationSpec,
    loaded: Mapping[str, Sequence[Scenario]],
) -> None:
    benchmark = manifest.get("benchmark")
    if not isinstance(benchmark, dict):
        _fail("ood_source", "$.benchmark", "expected an object")
    required_benchmark = {
        "benchmark_id",
        "manifest_name",
        "manifest_sha256",
        "split",
        "purpose",
        "scenario_hashes_sha256",
    }
    if set(benchmark) != required_benchmark:
        _fail(
            "ood_source",
            "$.benchmark",
            "benchmark binding keys differ from the frozen schema",
            details={
                "missing": sorted(required_benchmark - set(benchmark)),
                "unknown": sorted(set(benchmark) - required_benchmark),
            },
        )
    if benchmark.get("benchmark_id") != spec.id_slice["source"].get("benchmark_id"):
        _fail("ood_source", "$.benchmark.benchmark_id", "benchmark id mismatch")

    slices = _manifest_slice_map(manifest)
    id_entries = slices["id_validation"]["entries"]
    source_entries: list[dict[str, Any]] = []
    source_keys = {
        "source",
        "source_sha256",
        "scenario_id",
        "scenario_hash",
        "split_index",
    }
    for index, entry in enumerate(id_entries):
        provenance = entry.get("provenance")
        if (
            not isinstance(provenance, dict)
            or set(provenance) != {"kind", *source_keys}
            or provenance.get("kind") != "benchmark_validation"
        ):
            _fail(
                "ood_source",
                f"$.slices.id_validation.entries[{index}].provenance",
                "invalid benchmark provenance",
            )
        source_entries.append({key: provenance[key] for key in source_keys})
    _validate_source_binding(
        {**benchmark, "entries": source_entries},
        loaded["id_validation"],
    )

    for index, (entry, scenario) in enumerate(
        zip(
            slices["ood_size"]["entries"],
            loaded["ood_size"],
            strict=True,
        )
    ):
        if entry.get("provenance") != {
            "kind": "generated",
            "generator_seed": scenario.seed,
        }:
            _fail(
                "ood_source",
                f"$.slices.ood_size.entries[{index}].provenance",
                "generated scenario provenance mismatch",
            )

    for slice_id in ("ood_ccr", "ood_system"):
        for index, entry in enumerate(slices[slice_id]["entries"]):
            base = loaded["id_validation"][index]
            if entry.get("provenance") != {
                "kind": "deterministic_transform",
                "base_scenario_id": base.id,
                "base_scenario_hash": base.content_hash(),
            }:
                _fail(
                    "ood_source",
                    f"$.slices.{slice_id}.entries[{index}].provenance",
                    "transformed scenario provenance mismatch",
                )


def _materialized_scenarios(
    spec: _MaterializationSpec,
    validation_scenarios: Sequence[Scenario],
) -> dict[str, list[Scenario]]:
    if len(validation_scenarios) != spec.id_slice["scenario_count"]:
        _fail(
            "ood_source",
            "$validation_scenarios",
            "validation scenario count differs from the contract",
            details={
                "expected": spec.id_slice["scenario_count"],
                "actual": len(validation_scenarios),
            },
        )
    if len({scenario.id for scenario in validation_scenarios}) != len(
        validation_scenarios
    ) or len({scenario.content_hash() for scenario in validation_scenarios}) != len(
        validation_scenarios
    ):
        _fail("ood_source", "$validation_scenarios", "duplicate validation scenario")

    size_source = spec.size_slice["source"]
    size_scenarios = [
        generate_scenario(
            seed=seed,
            task_count=size_source["task_count"],
            resource_count=size_source["resource_count"],
            edge_probability=float(size_source["edge_probability"]),
            scenario_id=_scenario_id(
                size_source["scenario_id_template"],
                index,
                "$.slices.ood_size.source.scenario_id_template",
            ),
        )
        for index, seed in enumerate(size_source["generator_seeds"])
    ]
    ccr_source = spec.ccr_slice["source"]
    ccr_scenarios = [
        transform_ood_ccr(
            base,
            scenario_id=_scenario_id(
                ccr_source["scenario_id_template"],
                index,
                "$.slices.ood_ccr.source.scenario_id_template",
            ),
            edge_data_scale=float(ccr_source["edge_data_scale"]),
            off_diagonal_bandwidth_scale=float(
                ccr_source["off_diagonal_bandwidth_scale"]
            ),
        )
        for index, base in enumerate(validation_scenarios)
    ]
    system_source = spec.system_slice["source"]
    system_scenarios = [
        transform_ood_system(
            base,
            scenario_id=_scenario_id(
                system_source["scenario_id_template"],
                index,
                "$.slices.ood_system.source.scenario_id_template",
            ),
            resource_speed_multipliers=system_source["resource_speed_multipliers"],
            off_diagonal_bandwidth_scale=float(
                system_source["off_diagonal_bandwidth_scale"]
            ),
            off_diagonal_latency_scale=float(
                system_source["off_diagonal_latency_scale"]
            ),
        )
        for index, base in enumerate(validation_scenarios)
    ]
    result = {
        "id_validation": list(validation_scenarios),
        "ood_size": size_scenarios,
        "ood_ccr": ccr_scenarios,
        "ood_system": system_scenarios,
    }
    ids = [scenario.id for scenarios in result.values() for scenario in scenarios]
    hashes = [
        scenario.content_hash()
        for scenarios in result.values()
        for scenario in scenarios
    ]
    if len(ids) != len(set(ids)):
        _fail("ood_materialization", "$.slices", "scenario ids overlap across slices")
    if len(hashes) != len(set(hashes)):
        _fail(
            "ood_materialization", "$.slices", "scenario hashes overlap across slices"
        )
    return result


def _provenance(
    slice_id: str,
    index: int,
    scenario: Scenario,
    base_scenarios: Sequence[Scenario],
    source_entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if slice_id == "id_validation":
        return {"kind": "benchmark_validation", **dict(source_entries[index])}
    if slice_id == "ood_size":
        return {"kind": "generated", "generator_seed": scenario.seed}
    base = base_scenarios[index]
    return {
        "kind": "deterministic_transform",
        "base_scenario_id": base.id,
        "base_scenario_hash": base.content_hash(),
    }


def materialize_development_slices(
    contract_path: str | Path,
    validation_scenarios: Sequence[Scenario],
    source_binding: Mapping[str, Any],
    output_directory: str | Path,
) -> Path:
    """Atomically materialize the four frozen development slices without test access."""

    contract_source = Path(contract_path)
    try:
        contract = load_evaluation_contract(contract_source)
    except Exception as error:
        if isinstance(error, OODWorkflowError):
            raise
        _fail("ood_contract", "$contract", str(error))
    spec = _materialization_spec(contract)
    if source_binding.get("benchmark_id") != spec.id_slice["source"].get(
        "benchmark_id"
    ):
        _fail("ood_source", "$source_binding.benchmark_id", "benchmark id mismatch")
    source_entries = _validate_source_binding(source_binding, validation_scenarios)
    slices = _materialized_scenarios(spec, validation_scenarios)
    destination = Path(output_directory)
    if destination.exists():
        _fail(
            "ood_output_exists",
            "$output",
            "materialization destination must not already exist",
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}-staging-",
            dir=destination.parent,
        )
    )
    try:
        slice_reports: list[dict[str, Any]] = []
        contract_slices = _slice_map(contract)
        for slice_id in contract["modes"]["development"]:
            scenarios = slices[slice_id]
            entries: list[dict[str, Any]] = []
            for index, scenario in enumerate(scenarios):
                relative = PurePosixPath("slices", slice_id, f"{index:04d}.json")
                target = staging.joinpath(*relative.parts)
                _write_json(target, scenario.to_dict())
                entries.append(
                    {
                        "index": index,
                        "scenario_id": scenario.id,
                        "scenario_hash": scenario.content_hash(),
                        "file": relative.as_posix(),
                        "file_sha256": _file_sha256(target),
                        "seed": scenario.seed,
                        "task_count": scenario.task_count,
                        "resource_count": scenario.resource_count,
                        "edge_count": len(scenario.edges),
                        "provenance": _provenance(
                            slice_id,
                            index,
                            scenario,
                            slices["id_validation"],
                            source_entries,
                        ),
                    }
                )
            scenario_pairs = [
                {
                    "scenario_id": entry["scenario_id"],
                    "scenario_hash": entry["scenario_hash"],
                }
                for entry in entries
            ]
            slice_reports.append(
                {
                    "slice_id": slice_id,
                    "role": contract_slices[slice_id]["role"],
                    "scenario_count": len(entries),
                    "source": contract_slices[slice_id]["source"],
                    "scenario_set_sha256": canonical_json_sha256(scenario_pairs),
                    "entries": entries,
                }
            )
        manifest = {
            "format_version": 1,
            "mode": "p1_b02_development_slices",
            "materializer_version": MATERIALIZER_VERSION,
            "contract": {
                "contract_id": contract["contract_id"],
                "canonical_sha256": canonical_json_sha256(contract),
            },
            "benchmark": {
                key: value for key, value in source_binding.items() if key != "entries"
            },
            "test_accessed": False,
            "public_test_materialized": False,
            "slices": slice_reports,
        }
        manifest_path = staging / MATERIALIZATION_MANIFEST_NAME
        _write_json(manifest_path, manifest)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / MATERIALIZATION_MANIFEST_NAME


def materialize_p1_b02_ood(
    contract_path: str | Path,
    raw_benchmark_root: str | Path,
    benchmark_manifest_path: str | Path,
    output_directory: str | Path,
) -> Path:
    """Load only the verified validation split, then materialize ID/OOD slices."""

    benchmark_source = Path(benchmark_manifest_path)
    try:
        benchmark = load_benchmark_manifest(benchmark_source)
        validation = load_frozen_split(
            raw_benchmark_root,
            benchmark_source,
            "validation",
            purpose="model_selection",
        )
    except (BenchmarkValidationError, ScenarioValidationError, OSError) as error:
        _fail("ood_source", "$benchmark", str(error))
    validation_entries = [
        {
            "source": entry["source"],
            "source_sha256": entry["source_sha256"],
            "scenario_id": entry["scenario_id"],
            "scenario_hash": entry["scenario_hash"],
            "split_index": entry["split_index"],
        }
        for entry in benchmark["entries"]
        if entry["split"] == "validation"
    ]
    source_binding = {
        "benchmark_id": benchmark["benchmark_id"],
        "manifest_name": benchmark_source.name,
        "manifest_sha256": _file_sha256(benchmark_source),
        "split": "validation",
        "purpose": "model_selection",
        "scenario_hashes_sha256": benchmark["splits"]["validation"][
            "scenario_hashes_sha256"
        ],
        "entries": validation_entries,
    }
    return materialize_development_slices(
        contract_path,
        validation,
        source_binding,
        output_directory,
    )


def _manifest_slice_map(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw_slices = manifest.get("slices")
    if not isinstance(raw_slices, list) or any(
        not isinstance(item, dict) for item in raw_slices
    ):
        _fail("ood_manifest", "$.slices", "expected an array of objects")
    result = {item.get("slice_id"): item for item in raw_slices}
    if len(result) != len(raw_slices) or any(
        not isinstance(key, str) for key in result
    ):
        _fail("ood_manifest", "$.slices", "duplicate or invalid slice id")
    return result


def load_materialized_development_slices(
    contract_path: str | Path,
    materialization_root: str | Path,
    manifest_path: str | Path | None = None,
) -> tuple[dict[str, list[Scenario]], dict[str, Any]]:
    """Verify every materialized byte and deterministically reconstruct all slices."""

    contract_source = Path(contract_path)
    try:
        contract = load_evaluation_contract(contract_source)
    except Exception as error:
        _fail("ood_contract", "$contract", str(error))
    spec = _materialization_spec(contract)
    root = Path(materialization_root).resolve()
    source = Path(manifest_path or root / MATERIALIZATION_MANIFEST_NAME)
    manifest = _load_json(source, error_path="$manifest")
    if (
        manifest.get("format_version") != 1
        or manifest.get("mode") != "p1_b02_development_slices"
        or manifest.get("materializer_version") != MATERIALIZER_VERSION
    ):
        _fail("ood_manifest", "$manifest", "unsupported materialization manifest")
    if (
        manifest.get("test_accessed") is not False
        or manifest.get("public_test_materialized") is not False
    ):
        _fail("ood_test_gate", "$manifest.test_accessed", "public test is forbidden")
    expected_contract = {
        "contract_id": contract["contract_id"],
        "canonical_sha256": canonical_json_sha256(contract),
    }
    if manifest.get("contract") != expected_contract:
        _fail("ood_manifest", "$.contract", "contract binding mismatch")
    slices = _manifest_slice_map(manifest)
    if list(slices) != contract["modes"]["development"]:
        _fail("ood_manifest", "$.slices", "development slice order/set mismatch")
    contract_slices = _slice_map(contract)
    loaded: dict[str, list[Scenario]] = {}
    all_ids: set[str] = set()
    all_hashes: set[str] = set()
    for slice_id in contract["modes"]["development"]:
        item = slices[slice_id]
        if item.get("role") != contract_slices[slice_id]["role"]:
            _fail("ood_manifest", f"$.slices.{slice_id}.role", "slice role mismatch")
        if item.get("source") != contract_slices[slice_id]["source"]:
            _fail(
                "ood_manifest", f"$.slices.{slice_id}.source", "slice source mismatch"
            )
        entries = item.get("entries")
        expected_count = contract_slices[slice_id]["scenario_count"]
        if not isinstance(entries, list) or len(entries) != expected_count:
            _fail(
                "ood_manifest",
                f"$.slices.{slice_id}.entries",
                "scenario count mismatch",
            )
        scenarios: list[Scenario] = []
        scenario_pairs: list[dict[str, str]] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict) or entry.get("index") != index:
                _fail(
                    "ood_manifest",
                    f"$.slices.{slice_id}.entries[{index}]",
                    "invalid entry/index",
                )
            expected_relative = PurePosixPath("slices", slice_id, f"{index:04d}.json")
            if entry.get("file") != expected_relative.as_posix():
                _fail(
                    "ood_manifest",
                    f"$.slices.{slice_id}.entries[{index}].file",
                    "unexpected scenario path",
                )
            relative = PurePosixPath(entry["file"])
            if relative.is_absolute() or ".." in relative.parts:
                _fail(
                    "ood_manifest",
                    f"$.slices.{slice_id}.entries[{index}].file",
                    "unsafe path",
                )
            scenario_path = root.joinpath(*relative.parts).resolve()
            if not scenario_path.is_relative_to(root) or not scenario_path.is_file():
                _fail(
                    "ood_file",
                    f"$.slices.{slice_id}.entries[{index}].file",
                    "missing file",
                )
            if _file_sha256(scenario_path) != entry.get("file_sha256"):
                _fail(
                    "ood_file_hash",
                    f"$.slices.{slice_id}.entries[{index}].file_sha256",
                    "scenario file hash mismatch",
                )
            try:
                scenario = Scenario.load(scenario_path)
            except (ScenarioValidationError, OSError) as error:
                _fail(
                    "ood_scenario", f"$.slices.{slice_id}.entries[{index}]", str(error)
                )
            if scenario.id != entry.get("scenario_id"):
                _fail(
                    "ood_scenario",
                    f"$.slices.{slice_id}.entries[{index}].scenario_id",
                    "scenario id mismatch",
                )
            scenario_hash = scenario.content_hash()
            if scenario_hash != entry.get("scenario_hash"):
                _fail(
                    "ood_scenario",
                    f"$.slices.{slice_id}.entries[{index}].scenario_hash",
                    "scenario content hash mismatch",
                )
            for key, actual in (
                ("seed", scenario.seed),
                ("task_count", scenario.task_count),
                ("resource_count", scenario.resource_count),
                ("edge_count", len(scenario.edges)),
            ):
                if entry.get(key) != actual:
                    _fail(
                        "ood_scenario",
                        f"$.slices.{slice_id}.entries[{index}].{key}",
                        "scenario metadata mismatch",
                    )
            if scenario.id in all_ids or scenario_hash in all_hashes:
                _fail(
                    "ood_manifest",
                    f"$.slices.{slice_id}.entries[{index}]",
                    "cross-slice duplicate",
                )
            all_ids.add(scenario.id)
            all_hashes.add(scenario_hash)
            scenarios.append(scenario)
            scenario_pairs.append(
                {"scenario_id": scenario.id, "scenario_hash": scenario_hash}
            )
        if item.get("scenario_count") != len(scenarios) or item.get(
            "scenario_set_sha256"
        ) != canonical_json_sha256(scenario_pairs):
            _fail("ood_manifest", f"$.slices.{slice_id}", "slice aggregate mismatch")
        loaded[slice_id] = scenarios

    expected = _materialized_scenarios(spec, loaded["id_validation"])
    for slice_id in ("ood_size", "ood_ccr", "ood_system"):
        expected_pairs = [
            (scenario.id, scenario.content_hash()) for scenario in expected[slice_id]
        ]
        actual_pairs = [
            (scenario.id, scenario.content_hash()) for scenario in loaded[slice_id]
        ]
        if actual_pairs != expected_pairs:
            _fail(
                "ood_transform",
                f"$.slices.{slice_id}",
                "materialized scenarios differ from the frozen transform",
            )
    _validate_materialized_provenance(manifest, spec, loaded)
    return loaded, manifest


RunnerProvider = Callable[[str, int], SchedulerRunner]
Clock = Callable[[], float]


def _error_payload(error: Exception, policy: str, scenario: Scenario) -> dict[str, Any]:
    if isinstance(error, SchedulerAdapterError):
        return error.to_dict()
    if isinstance(error, TimeoutError):
        return {
            "code": "scheduler_timeout",
            "message": str(error) or "scheduler timed out",
            "scheduler": policy,
            "scenario_id": scenario.id,
        }
    return {
        "code": "scheduler_execution_failed",
        "message": str(error).strip() or type(error).__name__,
        "scheduler": policy,
        "scenario_id": scenario.id,
        "details": {"exception_type": type(error).__name__},
    }


def _schedule_only(
    runner: SchedulerRunner,
    scenario: Scenario,
    *,
    policy: str,
    clock: Clock,
) -> tuple[ScheduleResult | None, float, dict[str, Any] | None, int]:
    before = canonical_json_sha256(scenario.to_dict())
    illegal_action_count = 0
    start = clock()
    try:
        result = runner.schedule(scenario)
    except Exception as error:
        runtime_ms = (clock() - start) * 1000.0
        result = None
        payload = _error_payload(error, policy, scenario)
        if (
            isinstance(error, SchedulerAdapterError)
            and error.code == "scheduler_invalid_schedule"
        ):
            illegal_action_count = 1
    else:
        runtime_ms = (clock() - start) * 1000.0
        payload = None
    if not math.isfinite(runtime_ms) or runtime_ms < 0.0:
        _fail("ood_timing", "$.records.runtime_ms", "clock produced invalid runtime")
    if canonical_json_sha256(scenario.to_dict()) != before:
        _fail(
            "ood_input_mutation",
            "$.scenarios",
            "scheduler mutated a materialized scenario",
            details={"policy": policy, "scenario_id": scenario.id},
        )
    if result is not None:
        if not isinstance(result, ScheduleResult):
            payload = SchedulerAdapterError(
                "scheduler_invalid_response",
                "scheduler did not return a ScheduleResult",
                scheduler=policy,
                scenario_id=scenario.id,
                details={
                    "actual_type": (
                        f"{type(result).__module__}.{type(result).__qualname__}"
                    )
                },
            ).to_dict()
            result = None
        elif result.policy_name != policy:
            payload = SchedulerAdapterError(
                "scheduler_invalid_response",
                "scheduler returned a mismatched policy name",
                scheduler=policy,
                scenario_id=scenario.id,
                details={"actual_name": result.policy_name},
            ).to_dict()
            result = None
        else:
            try:
                # These validators are deliberately outside the measured interval.
                validate_schedule(scenario, result)
                validate_schedule_independent(scenario, result)
            except ValueError as error:
                payload = SchedulerAdapterError(
                    "scheduler_invalid_schedule",
                    "scheduler returned an invalid schedule",
                    scheduler=policy,
                    scenario_id=scenario.id,
                    details={"reason": str(error)},
                ).to_dict()
                result = None
                illegal_action_count = 1
    return result, runtime_ms, payload, illegal_action_count


def _runner_map(
    contract: Mapping[str, Any], provider: RunnerProvider
) -> dict[tuple[str, int], SchedulerRunner]:
    runners: dict[tuple[str, int], SchedulerRunner] = {}
    for policy in contract["policies"]:
        for seed in policy["required_seeds"]:
            try:
                runner = provider(policy["id"], seed)
            except Exception as error:
                _fail(
                    "ood_runner_provider",
                    f"$.runners.{policy['id']}.{seed}",
                    str(error).strip() or type(error).__name__,
                )
            if getattr(runner, "name", None) != policy["id"]:
                _fail(
                    "ood_runner_provider",
                    f"$.runners.{policy['id']}.{seed}",
                    "runner name does not match policy id",
                    details={"actual": getattr(runner, "name", None)},
                )
            runners[(policy["id"], seed)] = runner
    return runners


def produce_development_evidence(
    contract_path: str | Path,
    materialization_root: str | Path,
    runner_provider: RunnerProvider,
    output_path: str | Path,
    *,
    code: Mapping[str, Any],
    manifest_path: str | Path | None = None,
    clock: Clock = time.perf_counter,
) -> Path:
    """Run frozen development scenarios without training or public-test access."""

    contract_source = Path(contract_path)
    try:
        contract = load_evaluation_contract(contract_source)
    except Exception as error:
        _fail("ood_contract", "$contract", str(error))
    scenarios, materialization = load_materialized_development_slices(
        contract_source,
        materialization_root,
        manifest_path,
    )
    manifest_source = Path(
        manifest_path or Path(materialization_root) / MATERIALIZATION_MANIFEST_NAME
    )
    runners = _runner_map(contract, runner_provider)
    policies = contract["policies"]
    reference_policy = contract["reference_policy"]
    reference_seed = next(
        policy["required_seeds"][0]
        for policy in policies
        if policy["id"] == reference_policy
    )
    penalty = float(contract["failure_penalty_ratio"])
    records: list[dict[str, Any]] = []
    for slice_id in contract["modes"]["development"]:
        reference_results: dict[str, tuple[ScheduleResult, float]] = {}
        reference_runner = runners[(reference_policy, reference_seed)]
        for scenario in scenarios[slice_id]:
            result, runtime_ms, error, _ = _schedule_only(
                reference_runner,
                scenario,
                policy=reference_policy,
                clock=clock,
            )
            if result is None:
                _fail(
                    "ood_reference_failed",
                    "$.records",
                    "HEFT reference failed; ratios are undefined",
                    details={"scenario_id": scenario.id, "error": error},
                )
            reference_results[scenario.id] = (result, runtime_ms)

        for policy in policies:
            policy_id = policy["id"]
            for seed in policy["required_seeds"]:
                runner = runners[(policy_id, seed)]
                for scenario in scenarios[slice_id]:
                    heft_result, heft_runtime_ms = reference_results[scenario.id]
                    if policy_id == reference_policy:
                        result = heft_result
                        runtime_ms = heft_runtime_ms
                        error = None
                        illegal_action_count = 0
                    else:
                        (
                            result,
                            runtime_ms,
                            error,
                            illegal_action_count,
                        ) = _schedule_only(
                            runner,
                            scenario,
                            policy=policy_id,
                            clock=clock,
                        )
                    if result is None:
                        record = {
                            "status": "failure",
                            "ratio": None,
                            "score_ratio": penalty,
                            "penalty_applied": True,
                            "illegal_action_count": illegal_action_count,
                            "error_code": error["code"]
                            if error is not None
                            else "scheduler_execution_failed",
                        }
                    else:
                        ratio = float(result.makespan / heft_result.makespan)
                        record = {
                            "status": "success",
                            "ratio": ratio,
                            "score_ratio": ratio,
                            "penalty_applied": False,
                            "illegal_action_count": 0,
                            "error_code": None,
                        }
                    records.append(
                        {
                            "slice_id": slice_id,
                            "policy": policy_id,
                            "seed": seed,
                            "scenario_id": scenario.id,
                            "scenario_hash": scenario.content_hash(),
                            **record,
                            "runtime_ms": runtime_ms,
                        }
                    )
    slice_manifests = {
        item["slice_id"]: {
            "scenario_count": item["scenario_count"],
            "scenario_set_sha256": item["scenario_set_sha256"],
        }
        for item in materialization["slices"]
    }
    evidence = {
        "format_version": 1,
        "mode": "development",
        "contract_sha256": canonical_json_sha256(contract),
        "code": dict(code),
        "test_accessed": False,
        "slice_manifests": slice_manifests,
        "records": records,
        "records_sha256": canonical_json_sha256(records),
        "producer": {
            "format_version": 1,
            "mode": "p1_b02_read_only_development_evidence",
            "materialization_manifest_name": manifest_source.name,
            "materialization_manifest_sha256": _file_sha256(manifest_source),
            "runtime_scope": "wall_clock_runner_schedule_call_only",
            "validation_timing": "production_and_independent_validators_run_after_timer_stop",
            "training_started": False,
            "public_test_loaded": False,
        },
    }
    destination = Path(output_path)
    if destination.exists():
        _fail("ood_output_exists", "$output", "refusing to overwrite evidence output")
    _write_json_exclusive(destination, evidence)
    return destination
