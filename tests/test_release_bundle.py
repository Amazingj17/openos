from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from scripts.build_release_bundle import (
    MANIFEST_NAME,
    ReleaseBundleError,
    build_release_bundle,
    resolve_commit,
    verify_release_bundle,
)


ROOT = Path(__file__).resolve().parents[1]


def test_release_bundle_is_deterministic_source_only_and_self_verifying(
    tmp_path: Path,
) -> None:
    commit = resolve_commit(ROOT)
    first = build_release_bundle(ROOT, tmp_path / "first.zip", commit=commit)
    second = build_release_bundle(ROOT, tmp_path / "second.zip", commit=commit)

    assert (
        hashlib.sha256(first.read_bytes()).digest()
        == hashlib.sha256(second.read_bytes()).digest()
    )
    manifest = verify_release_bundle(first)
    assert manifest["source_commit"] == commit
    assert manifest["archive_policy"]["public_test_raw_bytes_included"] is False
    assert manifest["archive_policy"]["generated_outputs_included"] is False
    paths = [item["path"] for item in manifest["files"]]
    assert paths == sorted(paths)
    assert "README.md" in paths
    assert not any(path.startswith("outputs/") for path in paths)
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == [*paths, MANIFEST_NAME]
        stored = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
    assert stored == manifest


def test_release_bundle_rejects_a_tracked_secret_path(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "release-test@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Release Test"],
        cwd=repository,
        check=True,
    )
    (repository / "README.md").write_text("safe\n", encoding="utf-8")
    (repository / ".env").write_text("TOKEN=not-a-real-secret\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=repository, check=True)

    with pytest.raises(ReleaseBundleError, match="credential or private key"):
        build_release_bundle(repository, tmp_path / "unsafe.zip")
