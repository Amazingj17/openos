from __future__ import annotations

import hashlib
import json
import math
import re
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .scenario import Edge, Resource, Scenario, Task


BENCHMARK_ID = "stg-rnc50-hetero-trisched-v1"
PROJECTION_VERSION = 1
SPLIT_SALT = "trisched-p1-b01-stg-rnc50-v1"
DEFAULT_SPLIT_COUNTS = {"train": 120, "validation": 30, "test": 30}
SPLIT_ORDER = ("train", "validation", "test")
SPLIT_PURPOSES = {
    "teacher": frozenset({"train"}),
    "training": frozenset({"train"}),
    "model_selection": frozenset({"validation"}),
    "evaluation": frozenset({"validation", "test"}),
}

SOURCE_RECORD_ID = 18_927_122
SOURCE_DOI = "10.5281/zenodo.18927122"
SOURCE_ARCHIVE_NAME = "rnc50_hetero_json.tar.xz"
SOURCE_ARCHIVE_URL = (
    "https://zenodo.org/api/records/18927122/files/"
    "rnc50_hetero_json.tar.xz/content"
)
SOURCE_ARCHIVE_BYTES = 125_500
SOURCE_ARCHIVE_MD5 = "0d1a6b83935c31dd003172cccdbab933"
SOURCE_ARCHIVE_SHA256 = (
    "03bc163c13ae8601f8cb20ac1573746a5262c50edbf0c4e9e748968675ea5f7d"
)
SOURCE_LICENSE_SHA256 = (
    "bb96c8d739221ae0273ab1444f1a0d8200b82eca6ba4863f3eb79585889e7b4e"
)
UPSTREAM_REPOSITORY_COMMIT = "22134e74028b0164032fbcc4c806bfa7c385e551"

_TASK_NAME = re.compile(r"T([1-9][0-9]*)")
_SOURCE_NAME = re.compile(r"rand([0-9]{4})_hetero[.]json")

_RESOURCES = (
    Resource(0, "stg-device-0", "device", 1.0),
    Resource(1, "stg-edge-0", "edge", 2.4),
    Resource(2, "stg-cloud-0", "cloud", 4.8),
)
_BANDWIDTH_MBPS = (
    (1e9, 100.0, 100.0),
    (100.0, 1e9, 10_000.0),
    (100.0, 10_000.0, 1e9),
)
_LATENCY_SECONDS = (
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
)


class BenchmarkValidationError(ValueError):
    """Stable validation error for public benchmark inputs and manifests."""

    def __init__(self, code: str, path: str, message: str) -> None:
        self.code = code
        self.path = path
        self.message = message
        super().__init__(f"{code} at {path}: {message}")

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path, "message": self.message}


def _fail(code: str, path: str, message: str) -> None:
    raise BenchmarkValidationError(code, path, message)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        _fail("type_error", path, "expected an object with string keys")
    return value


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail("type_error", path, "expected an array")
    return value


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("type_error", path, "expected a number")
    result = float(value)
    if not math.isfinite(result):
        _fail("non_finite", path, "number must be finite")
    return result


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("type_error", path, "expected an integer")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
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
    return _sha256_bytes(encoded)


def projection_metadata() -> dict[str, Any]:
    return {
        "version": PROJECTION_VERSION,
        "scope": "STG topology/duration/data projected into TriSched static model",
        "task_workload": "upstream tasks.<id>.duration",
        "edge_data_mb": "source/predecessor task data",
        "ignored_upstream_fields": [
            "cores",
            "memory_required",
            "features",
            "tags",
            "system_config",
        ],
        "capability_constraints_preserved": False,
        "resources": [
            {
                "id": item.id,
                "name": item.name,
                "kind": item.kind,
                "speed": item.speed,
            }
            for item in _RESOURCES
        ],
        "bandwidth_unit": "MB/s",
        "bandwidth": [list(row) for row in _BANDWIDTH_MBPS],
        "latency_unit": "seconds",
        "latency": [list(row) for row in _LATENCY_SECONDS],
    }


def source_metadata() -> dict[str, Any]:
    return {
        "title": (
            "Standard Task Graph (STG) Dataset With JSON Conversions for "
            "Workflow Scheduling in Heterogeneous HPC Systems"
        ),
        "record_id": SOURCE_RECORD_ID,
        "doi": SOURCE_DOI,
        "record_url": f"https://zenodo.org/records/{SOURCE_RECORD_ID}",
        "archive": {
            "name": SOURCE_ARCHIVE_NAME,
            "url": SOURCE_ARCHIVE_URL,
            "bytes": SOURCE_ARCHIVE_BYTES,
            "md5": SOURCE_ARCHIVE_MD5,
            "sha256": SOURCE_ARCHIVE_SHA256,
        },
        "license": {
            "spdx": "CC-BY-4.0",
            "license_file_sha256": SOURCE_LICENSE_SHA256,
            "attribution_required": True,
            "original_stg_credit": (
                "Hiroshi Kasahara and collaborators, Waseda University"
            ),
        },
        "repository": {
            "url": (
                "https://github.com/AasishKumarSharma/grapheonrl-benchmark"
            ),
            "commit": UPSTREAM_REPOSITORY_COMMIT,
            "root_license_status": "missing; no code copied",
        },
        "excluded": [
            "upstream solver code (no root license)",
            "Zenodo record 20419279 results (rights not declared)",
        ],
    }


def split_policy_metadata(counts: Mapping[str, int]) -> dict[str, Any]:
    return {
        "salt": SPLIT_SALT,
        "rank_key": "sha256(utf8(salt + NUL + source_sha256))",
        "order": list(SPLIT_ORDER),
        "counts": dict(counts),
        "selection": "ascending rank_key; source path is the tie-break",
        "test_is_model_selection_forbidden": True,
    }


def _load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()

    def reject_constant(value: str) -> None:
        _fail("non_finite", "$", f"JSON constant {value} is not allowed")

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        _fail("encoding_error", "$", f"expected UTF-8: {error.reason}")
    try:
        payload = json.loads(text, parse_constant=reject_constant)
    except json.JSONDecodeError as error:
        _fail(
            "json_syntax",
            "$",
            f"line {error.lineno}, column {error.colno}: {error.msg}",
        )
    return dict(_mapping(payload, "$")), raw


def scenario_id_for_source(path: str | Path) -> str:
    name = Path(path).name
    match = _SOURCE_NAME.fullmatch(name)
    if match is None:
        _fail(
            "source_name",
            "$source",
            "expected randNNNN_hetero.json",
        )
    return f"stg-rnc50-hetero-rand{match.group(1)}"


def load_stg_json(
    source: str | Path,
    *,
    scenario_id: str | None = None,
) -> Scenario:
    """Load one pinned STG JSON file through projection version 1.

    This is deliberately a capability-relaxed topology projection. It does not
    claim to reproduce the upstream CPU/GPU/core/memory scheduling model.
    """

    path = Path(source)
    payload, _ = _load_json(path)
    meta = _mapping(payload.get("meta"), "$.meta")
    stg_info = _mapping(meta.get("stg_info"), "$.meta.stg_info")
    seed = _integer(stg_info.get("random_seed"), "$.meta.stg_info.random_seed")
    task_payload = _mapping(payload.get("tasks"), "$.tasks")
    if not task_payload:
        _fail("value_error", "$.tasks", "at least one task is required")

    numbered: list[tuple[int, str, Mapping[str, Any]]] = []
    for name, value in task_payload.items():
        match = _TASK_NAME.fullmatch(name)
        if match is None:
            _fail("task_name", f"$.tasks.{name}", "expected T followed by an integer")
        numbered.append((int(match.group(1)), name, _mapping(value, f"$.tasks.{name}")))
    numbered.sort()
    expected_numbers = list(range(1, len(numbered) + 1))
    actual_numbers = [number for number, _, _ in numbered]
    if actual_numbers != expected_numbers:
        _fail("task_ids", "$.tasks", "task names must be contiguous from T1")

    task_id = {name: number - 1 for number, name, _ in numbered}
    workloads: dict[str, float] = {}
    output_data: dict[str, float] = {}
    dependencies: dict[str, tuple[str, ...]] = {}
    for _, name, item in numbered:
        duration = _finite_number(item.get("duration"), f"$.tasks.{name}.duration")
        if duration <= 0:
            _fail("value_error", f"$.tasks.{name}.duration", "must be positive")
        data = _finite_number(item.get("data"), f"$.tasks.{name}.data")
        if data < 0:
            _fail("value_error", f"$.tasks.{name}.data", "must be non-negative")
        raw_dependencies = _sequence(
            item.get("dependencies"), f"$.tasks.{name}.dependencies"
        )
        values: list[str] = []
        for index, dependency in enumerate(raw_dependencies):
            if not isinstance(dependency, str):
                _fail(
                    "type_error",
                    f"$.tasks.{name}.dependencies[{index}]",
                    "expected a task name",
                )
            if dependency not in task_id:
                _fail(
                    "unknown_dependency",
                    f"$.tasks.{name}.dependencies[{index}]",
                    f"unknown task {dependency!r}",
                )
            if dependency == name:
                _fail(
                    "self_dependency",
                    f"$.tasks.{name}.dependencies[{index}]",
                    "a task cannot depend on itself",
                )
            values.append(dependency)
        if len(values) != len(set(values)):
            _fail(
                "duplicate_dependency",
                f"$.tasks.{name}.dependencies",
                "dependencies must be unique",
            )
        workloads[name] = duration
        output_data[name] = data
        dependencies[name] = tuple(values)

    tasks = tuple(
        Task(id=task_id[name], workload=workloads[name])
        for _, name, _ in numbered
    )
    edges = tuple(
        Edge(
            source=task_id[parent],
            target=task_id[name],
            data=output_data[parent],
        )
        for _, name, _ in numbered
        for parent in sorted(dependencies[name], key=lambda item: task_id[item])
    )
    return Scenario(
        id=scenario_id or scenario_id_for_source(path),
        seed=seed,
        tasks=tasks,
        resources=_RESOURCES,
        edges=edges,
        bandwidth=_BANDWIDTH_MBPS,
        latency=_LATENCY_SECONDS,
    )


def build_stg_manifest(
    source_directory: str | Path,
    split_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Build the deterministic manifest; committed output must be reviewed."""

    source_root = Path(source_directory)
    counts = dict(split_counts or DEFAULT_SPLIT_COUNTS)
    if tuple(counts) != SPLIT_ORDER or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in counts.values()
    ):
        raise ValueError(f"split_counts must contain positive {SPLIT_ORDER}")
    paths = sorted(source_root.glob("rand[0-9][0-9][0-9][0-9]_hetero.json"))
    if len(paths) != sum(counts.values()):
        raise ValueError(
            f"expected {sum(counts.values())} source files, found {len(paths)}"
        )

    candidates: list[dict[str, Any]] = []
    for path in paths:
        scenario_id = scenario_id_for_source(path)
        scenario = load_stg_json(path, scenario_id=scenario_id)
        source_sha256 = _sha256_file(path)
        rank_key = _sha256_bytes(
            f"{SPLIT_SALT}\0{source_sha256}".encode("utf-8")
        )
        candidates.append(
            {
                "source": f"rnc50_hetero/{path.name}",
                "source_sha256": source_sha256,
                "rank_key": rank_key,
                "scenario_id": scenario.id,
                "scenario_hash": scenario.content_hash(),
                "task_count": scenario.task_count,
                "edge_count": len(scenario.edges),
            }
        )
    candidates.sort(key=lambda item: (item["rank_key"], item["source"]))

    entries: list[dict[str, Any]] = []
    offset = 0
    for split in SPLIT_ORDER:
        for split_index, item in enumerate(candidates[offset : offset + counts[split]]):
            entries.append({"split": split, "split_index": split_index, **item})
        offset += counts[split]

    split_metadata = {}
    for split in SPLIT_ORDER:
        hashes = [
            entry["scenario_hash"] for entry in entries if entry["split"] == split
        ]
        split_metadata[split] = {
            "count": len(hashes),
            "scenario_hashes_sha256": _json_sha256(hashes),
        }

    return {
        "format_version": 1,
        "benchmark_id": BENCHMARK_ID,
        "source": source_metadata(),
        "projection": projection_metadata(),
        "split_policy": split_policy_metadata(counts),
        "splits": split_metadata,
        "entries": entries,
    }


def load_benchmark_manifest(path: str | Path) -> dict[str, Any]:
    payload, _ = _load_json(Path(path))
    if payload.get("format_version") != 1:
        _fail("manifest_version", "$.format_version", "expected 1")
    if payload.get("benchmark_id") != BENCHMARK_ID:
        _fail("benchmark_id", "$.benchmark_id", f"expected {BENCHMARK_ID}")
    source = _mapping(payload.get("source"), "$.source")
    if dict(source) != source_metadata():
        _fail("source_metadata", "$.source", "pinned source/license changed")
    if payload.get("projection") != projection_metadata():
        _fail("projection", "$.projection", "projection contract changed")

    split_policy = _mapping(payload.get("split_policy"), "$.split_policy")
    counts = _mapping(split_policy.get("counts"), "$.split_policy.counts")
    if tuple(counts) != SPLIT_ORDER:
        _fail("split_names", "$.split_policy.counts", "unexpected split order")
    normalized_counts = {
        name: _integer(counts[name], f"$.split_policy.counts.{name}")
        for name in SPLIT_ORDER
    }
    if any(value <= 0 for value in normalized_counts.values()):
        _fail("split_count", "$.split_policy.counts", "counts must be positive")
    if dict(split_policy) != split_policy_metadata(normalized_counts):
        _fail("split_policy", "$.split_policy", "frozen split policy changed")
    entries = _sequence(payload.get("entries"), "$.entries")
    if len(entries) != sum(normalized_counts.values()):
        _fail("entry_count", "$.entries", "entry count does not match split counts")

    sources: set[str] = set()
    source_hashes: set[str] = set()
    scenario_hashes: set[str] = set()
    scenario_ids: set[str] = set()
    rank_keys: list[str] = []
    split_hashes: dict[str, list[str]] = {name: [] for name in SPLIT_ORDER}
    split_indices: dict[str, list[int]] = {name: [] for name in SPLIT_ORDER}
    for index, raw_entry in enumerate(entries):
        entry = _mapping(raw_entry, f"$.entries[{index}]")
        split = entry.get("split")
        if split not in SPLIT_ORDER:
            _fail("split_name", f"$.entries[{index}].split", "unknown split")
        source_path = entry.get("source")
        source_sha = entry.get("source_sha256")
        scenario_hash = entry.get("scenario_hash")
        rank_key = entry.get("rank_key")
        scenario_id = entry.get("scenario_id")
        if not all(
            isinstance(value, str)
            for value in (
                source_path,
                source_sha,
                scenario_hash,
                rank_key,
                scenario_id,
            )
        ):
            _fail(
                "type_error",
                f"$.entries[{index}]",
                "hash/path fields must be strings",
            )
        relative = PurePosixPath(source_path)
        if (
            relative.is_absolute()
            or len(relative.parts) != 2
            or relative.parts[0] != "rnc50_hetero"
            or "\\" in source_path
        ):
            _fail("source_path", f"$.entries[{index}].source", source_path)
        expected_scenario_id = scenario_id_for_source(relative.name)
        if scenario_id != expected_scenario_id:
            _fail(
                "scenario_id",
                f"$.entries[{index}].scenario_id",
                f"expected {expected_scenario_id}",
            )
        for field, value in (
            ("source_sha256", source_sha),
            ("scenario_hash", scenario_hash),
            ("rank_key", rank_key),
        ):
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                _fail("hash_format", f"$.entries[{index}].{field}", value)
        expected_rank = _sha256_bytes(
            f"{SPLIT_SALT}\0{source_sha}".encode("utf-8")
        )
        if rank_key != expected_rank:
            _fail("rank_key", f"$.entries[{index}].rank_key", "rank differs")
        if source_path in sources or source_sha in source_hashes:
            _fail("source_duplicate", f"$.entries[{index}]", "source is duplicated")
        if scenario_hash in scenario_hashes:
            _fail(
                "scenario_duplicate",
                f"$.entries[{index}]",
                "scenario hash is duplicated",
            )
        if scenario_id in scenario_ids:
            _fail("scenario_id_duplicate", f"$.entries[{index}]", scenario_id)
        sources.add(source_path)
        source_hashes.add(source_sha)
        scenario_hashes.add(scenario_hash)
        scenario_ids.add(scenario_id)
        rank_keys.append(rank_key)
        task_count = _integer(entry.get("task_count"), f"$.entries[{index}].task_count")
        edge_count = _integer(entry.get("edge_count"), f"$.entries[{index}].edge_count")
        if task_count <= 0 or edge_count < 0:
            _fail("scenario_shape", f"$.entries[{index}]", "invalid task/edge count")
        split_hashes[split].append(scenario_hash)
        split_indices[split].append(
            _integer(entry.get("split_index"), f"$.entries[{index}].split_index")
        )

    split_metadata = _mapping(payload.get("splits"), "$.splits")
    for split in SPLIT_ORDER:
        if split_indices[split] != list(range(normalized_counts[split])):
            _fail(
                "split_order",
                f"$.entries[{split}]",
                "split indices are not contiguous",
            )
        metadata = _mapping(split_metadata.get(split), f"$.splits.{split}")
        expected = {
            "count": normalized_counts[split],
            "scenario_hashes_sha256": _json_sha256(split_hashes[split]),
        }
        if dict(metadata) != expected:
            _fail("split_hash", f"$.splits.{split}", "split aggregate hash differs")
    if rank_keys != sorted(rank_keys):
        _fail("rank_order", "$.entries", "entries are not in frozen rank order")
    return payload


def _load_verified_entry(
    root: Path,
    entry: Mapping[str, Any],
    index: int,
) -> Scenario:
    relative = PurePosixPath(entry["source"])
    if relative.is_absolute() or ".." in relative.parts:
        _fail("unsafe_path", f"$.entries[{index}].source", "path escapes root")
    source = root.joinpath(*relative.parts).resolve()
    if not source.is_relative_to(root) or not source.is_file():
        _fail("source_missing", f"$.entries[{index}].source", str(source))
    if _sha256_file(source) != entry["source_sha256"]:
        _fail("source_hash", f"$.entries[{index}].source_sha256", str(source))
    scenario = load_stg_json(source, scenario_id=entry["scenario_id"])
    if scenario.content_hash() != entry["scenario_hash"]:
        _fail("scenario_hash", f"$.entries[{index}].scenario_hash", str(source))
    if (
        scenario.task_count != entry["task_count"]
        or len(scenario.edges) != entry["edge_count"]
    ):
        _fail("scenario_shape", f"$.entries[{index}]", str(source))
    return scenario


def load_frozen_split(
    extracted_root: str | Path,
    manifest_path: str | Path,
    split: str,
    *,
    purpose: str,
) -> list[Scenario]:
    """Load one verified split through an explicit anti-leakage purpose gate."""

    if split not in SPLIT_ORDER:
        _fail("split_name", "$split", f"unknown split {split!r}")
    allowed = SPLIT_PURPOSES.get(purpose)
    if allowed is None:
        _fail("split_purpose", "$purpose", f"unknown purpose {purpose!r}")
    if split not in allowed:
        _fail(
            "split_usage",
            "$split",
            f"split {split!r} is forbidden for purpose {purpose!r}",
        )
    root = Path(extracted_root).resolve()
    manifest = load_benchmark_manifest(manifest_path)
    return [
        _load_verified_entry(root, entry, index)
        for index, entry in enumerate(manifest["entries"])
        if entry["split"] == split
    ]


def verify_frozen_splits(
    extracted_root: str | Path,
    manifest_path: str | Path,
) -> dict[str, list[Scenario]]:
    root = Path(extracted_root).resolve()
    manifest = load_benchmark_manifest(manifest_path)
    splits: dict[str, list[Scenario]] = {name: [] for name in SPLIT_ORDER}
    for index, entry in enumerate(manifest["entries"]):
        splits[entry["split"]].append(
            _load_verified_entry(root, entry, index)
        )
    return splits


def verify_archive(
    archive_path: str | Path,
    *,
    expected_bytes: int = SOURCE_ARCHIVE_BYTES,
    expected_sha256: str = SOURCE_ARCHIVE_SHA256,
) -> None:
    path = Path(archive_path)
    if not path.is_file():
        _fail("archive_missing", "$archive", str(path))
    if path.stat().st_size != expected_bytes:
        _fail("archive_size", "$archive", f"expected {expected_bytes} bytes")
    if _sha256_file(path) != expected_sha256:
        _fail("archive_hash", "$archive", "SHA-256 differs from pinned source")


def extract_verified_archive(
    archive_path: str | Path,
    destination: str | Path,
    expected_files: Sequence[str],
    *,
    expected_bytes: int = SOURCE_ARCHIVE_BYTES,
    expected_sha256: str = SOURCE_ARCHIVE_SHA256,
) -> None:
    verify_archive(
        archive_path,
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
    )
    output = Path(destination)
    if output.exists() and any(output.iterdir()):
        _fail("destination_not_empty", "$destination", str(output))
    output.mkdir(parents=True, exist_ok=True)
    expected = set(expected_files)
    with tarfile.open(archive_path, mode="r:xz") as archive:
        members = archive.getmembers()
        actual_files: set[str] = set()
        for member in members:
            relative = PurePosixPath(member.name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "\\" in member.name
            ):
                _fail("unsafe_archive_path", "$archive", member.name)
            if member.issym() or member.islnk() or not (
                member.isdir() or member.isfile()
            ):
                _fail("unsafe_archive_type", "$archive", member.name)
            target = output.resolve().joinpath(*relative.parts).resolve()
            if not target.is_relative_to(output.resolve()):
                _fail("unsafe_archive_path", "$archive", member.name)
            if member.isfile():
                normalized = relative.as_posix()
                if normalized in actual_files:
                    _fail("archive_duplicate", "$archive", normalized)
                actual_files.add(normalized)
        if actual_files != expected:
            _fail(
                "archive_members",
                "$archive",
                f"expected {len(expected)} files, found {len(actual_files)}",
            )
        for member in members:
            archive.extract(member, path=output, set_attrs=False)
