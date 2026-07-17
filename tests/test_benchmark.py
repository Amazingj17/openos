from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from trisched.benchmark import (
    BenchmarkValidationError,
    build_stg_manifest,
    extract_verified_archive,
    load_benchmark_manifest,
    load_stg_json,
    verify_frozen_splits,
)


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "benchmark"
    / "stg_projection_example.json"
)
REPOSITORY = Path(__file__).resolve().parents[1]
FROZEN_MANIFEST = (
    REPOSITORY / "data" / "benchmarks" / "stg-rnc50-hetero-v1.json"
)


def test_stg_projection_has_explicit_source_data_semantics() -> None:
    scenario = load_stg_json(FIXTURE, scenario_id="fixture")
    assert scenario.seed == 41
    assert [task.workload for task in scenario.tasks] == [4.0, 2.0, 5.0]
    assert [(item.kind, item.speed) for item in scenario.resources] == [
        ("device", 1.0),
        ("edge", 2.4),
        ("cloud", 4.8),
    ]
    assert [(edge.source, edge.target, edge.data) for edge in scenario.edges] == [
        (0, 1, 6.0),
        (0, 2, 6.0),
        (1, 2, 3.0),
    ]
    assert scenario.bandwidth[0][1] == 100.0
    assert scenario.bandwidth[1][2] == 10_000.0


def test_stg_projection_rejects_duplicate_and_unknown_dependencies(
    tmp_path: Path,
) -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["tasks"]["T2"]["dependencies"] = ["T1", "T1"]
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BenchmarkValidationError) as caught:
        load_stg_json(duplicate, scenario_id="duplicate")
    assert caught.value.code == "duplicate_dependency"

    payload["tasks"]["T2"]["dependencies"] = ["T99"]
    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BenchmarkValidationError) as caught:
        load_stg_json(unknown, scenario_id="unknown")
    assert caught.value.code == "unknown_dependency"


def _write_source_set(root: Path, count: int = 6) -> None:
    base = json.loads(FIXTURE.read_text(encoding="utf-8"))
    root.mkdir(parents=True)
    for index in range(count):
        payload = json.loads(json.dumps(base))
        payload["meta"]["stg_info"]["random_seed"] = 100 + index
        payload["tasks"]["T1"]["duration"] = 4.0 + index
        (root / f"rand{index:04d}_hetero.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def test_manifest_freezes_disjoint_hashes_and_detects_source_change(
    tmp_path: Path,
) -> None:
    extracted = tmp_path / "raw"
    source = extracted / "rnc50_hetero"
    _write_source_set(source)
    manifest = build_stg_manifest(
        source, {"train": 4, "validation": 1, "test": 1}
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    splits = verify_frozen_splits(extracted, manifest_path)
    assert {name: len(items) for name, items in splits.items()} == {
        "train": 4,
        "validation": 1,
        "test": 1,
    }
    hashes = {
        name: {scenario.content_hash() for scenario in items}
        for name, items in splits.items()
    }
    assert not (hashes["train"] & hashes["validation"])
    assert not (hashes["train"] & hashes["test"])
    assert not (hashes["validation"] & hashes["test"])

    changed = extracted / manifest["entries"][0]["source"]
    changed.write_text(changed.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(BenchmarkValidationError) as caught:
        verify_frozen_splits(extracted, manifest_path)
    assert caught.value.code == "source_hash"

    manifest["source"]["license"]["spdx"] = "MIT"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BenchmarkValidationError) as caught:
        load_benchmark_manifest(manifest_path)
    assert caught.value.code == "source_metadata"


def test_archive_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.xz"
    content = b"{}"
    with tarfile.open(archive, mode="w:xz") as handle:
        member = tarfile.TarInfo("../escaped.json")
        member.size = len(content)
        handle.addfile(member, io.BytesIO(content))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(BenchmarkValidationError) as caught:
        extract_verified_archive(
            archive,
            tmp_path / "extract",
            ["rnc50_hetero/rand0000_hetero.json"],
            expected_bytes=archive.stat().st_size,
            expected_sha256=digest,
        )
    assert caught.value.code == "unsafe_archive_path"
    assert not (tmp_path / "escaped.json").exists()


def test_committed_manifest_is_internally_consistent() -> None:
    manifest = load_benchmark_manifest(FROZEN_MANIFEST)
    assert manifest["split_policy"]["counts"] == {
        "train": 120,
        "validation": 30,
        "test": 30,
    }
    assert len(manifest["entries"]) == 180
    assert all(entry["task_count"] == 50 for entry in manifest["entries"])
    assert len({entry["source_sha256"] for entry in manifest["entries"]}) == 180
    assert len({entry["scenario_hash"] for entry in manifest["entries"]}) == 180
