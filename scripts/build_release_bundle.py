from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


MANIFEST_NAME = "RELEASE_MANIFEST.json"
_FORBIDDEN_PREFIXES = (".git/", "outputs/", "data/raw/", "data/downloads/")
_SECRET_BASENAMES = {
    ".env",
    "id_dsa",
    "id_ed25519",
    "id_rsa",
    "credentials.json",
    "public_test_authorization.json",
}
_SECRET_SUFFIXES = {".key", ".p12", ".pfx", ".pem"}
_SECRET_MARKERS = (
    b"-----BEGIN " + b"PRIVATE KEY-----",
    b"-----BEGIN " + b"OPENSSH PRIVATE KEY-----",
    b"-----BEGIN " + b"RSA PRIVATE KEY-----",
)
_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


class ReleaseBundleError(RuntimeError):
    pass


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _run_git(
    repository: Path,
    arguments: Sequence[str],
    *,
    text: bool = False,
) -> subprocess.CompletedProcess[Any]:
    try:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=text,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ReleaseBundleError(f"git command failed: {error}") from error


def resolve_commit(repository: str | Path, reference: str = "HEAD") -> str:
    root = Path(repository).resolve()
    result = _run_git(
        root, ["rev-parse", "--verify", f"{reference}^{{commit}}"], text=True
    )
    commit = result.stdout.strip()
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ReleaseBundleError("git did not return a full lowercase commit id")
    return commit


def _tracked_blobs(repository: Path, commit: str) -> list[tuple[str, str]]:
    output = _run_git(repository, ["ls-tree", "-r", "-z", commit]).stdout
    entries: list[tuple[str, str]] = []
    for raw_record in output.split(b"\0"):
        if not raw_record:
            continue
        try:
            metadata, raw_path = raw_record.split(b"\t", 1)
            mode, object_type, raw_object_id = metadata.split(b" ", 2)
            path = raw_path.decode("utf-8")
            object_id = raw_object_id.decode("ascii")
        except (ValueError, UnicodeDecodeError) as error:
            raise ReleaseBundleError(
                "git tree contains an unsupported entry"
            ) from error
        if object_type != b"blob" or mode == b"120000":
            raise ReleaseBundleError(
                f"release rejects non-regular tracked entry: {path}"
            )
        entries.append((path, object_id))
    return sorted(entries)


def _path_rejection(path: str) -> str | None:
    normalized = PurePosixPath(path).as_posix()
    parts = PurePosixPath(normalized).parts
    if not parts or normalized.startswith("/") or ".." in parts:
        return "path is not a safe repository-relative POSIX path"
    lowered = normalized.lower()
    if any(lowered.startswith(prefix) for prefix in _FORBIDDEN_PREFIXES):
        return "path belongs to a forbidden generated/raw-data prefix"
    basename = parts[-1].lower()
    if (
        basename in _SECRET_BASENAMES
        or PurePosixPath(basename).suffix in _SECRET_SUFFIXES
    ):
        return "path looks like a credential or private key"
    return None


def _blob_bytes(repository: Path, object_id: str) -> bytes:
    return _run_git(repository, ["cat-file", "blob", object_id]).stdout


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=_FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def build_release_bundle(
    repository: str | Path,
    output: str | Path,
    *,
    commit: str = "HEAD",
) -> Path:
    """Build a deterministic source-only zip from immutable Git blobs."""

    root = Path(repository).resolve()
    resolved_commit = resolve_commit(root, commit)
    target = Path(output).resolve()
    if target.exists():
        raise ReleaseBundleError(f"refusing to overwrite release bundle: {target}")

    payloads: list[tuple[str, bytes]] = []
    file_records: list[dict[str, Any]] = []
    for path, object_id in _tracked_blobs(root, resolved_commit):
        rejection = _path_rejection(path)
        if rejection is not None:
            raise ReleaseBundleError(f"unsafe tracked path {path!r}: {rejection}")
        payload = _blob_bytes(root, object_id)
        if any(marker in payload for marker in _SECRET_MARKERS):
            raise ReleaseBundleError(
                f"private-key marker found in tracked file: {path}"
            )
        payloads.append((path, payload))
        file_records.append(
            {
                "path": path,
                "bytes": len(payload),
                "sha256": _sha256(payload),
                "git_blob": object_id,
            }
        )

    manifest = {
        "format_version": 1,
        "bundle_type": "trisched_source_release",
        "source_commit": resolved_commit,
        "path_convention": "repository-relative POSIX paths",
        "archive_policy": {
            "source": "immutable Git blobs",
            "entry_order": "UTF-8 path sort",
            "entry_timestamp": "1980-01-01T00:00:00",
            "public_test_raw_bytes_included": False,
            "generated_outputs_included": False,
            "credentials_included": False,
        },
        "files": file_records,
    }
    manifest_bytes = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(
            target,
            mode="x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for path, payload in payloads:
                archive.writestr(_zip_info(path), payload)
            archive.writestr(_zip_info(MANIFEST_NAME), manifest_bytes)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        target.unlink(missing_ok=True)
        raise ReleaseBundleError(f"cannot write release bundle: {error}") from error
    return target


def verify_release_bundle(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    try:
        with zipfile.ZipFile(source, "r") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or MANIFEST_NAME not in names:
                raise ReleaseBundleError(
                    "release zip has duplicate names or no manifest"
                )
            manifest = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
            expected_names = [item["path"] for item in manifest["files"]]
            if names != [*expected_names, MANIFEST_NAME]:
                raise ReleaseBundleError(
                    "release zip entry order/content differs from manifest"
                )
            for item in manifest["files"]:
                payload = archive.read(item["path"])
                if len(payload) != item["bytes"] or _sha256(payload) != item["sha256"]:
                    raise ReleaseBundleError(
                        f"release payload hash mismatch: {item['path']}"
                    )
    except (
        OSError,
        KeyError,
        TypeError,
        UnicodeError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ) as error:
        if isinstance(error, ReleaseBundleError):
            raise
        raise ReleaseBundleError(f"cannot verify release bundle: {error}") from error
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or verify a deterministic TriSched source bundle"
    )
    parser.add_argument("--repository", default=".")
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--output", default=None)
    parser.add_argument("--verify", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.verify is not None:
            manifest = verify_release_bundle(args.verify)
            print(
                json.dumps(
                    {
                        "ok": True,
                        "path": str(Path(args.verify).resolve()),
                        "source_commit": manifest["source_commit"],
                        "file_count": len(manifest["files"]),
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        repository = Path(args.repository).resolve()
        commit = resolve_commit(repository, args.commit)
        output = (
            Path(args.output)
            if args.output is not None
            else repository
            / "outputs"
            / "release"
            / f"trisched-source-{commit[:12]}.zip"
        )
        result = build_release_bundle(repository, output, commit=commit)
        print(result)
        print(f"sha256={_sha256(result.read_bytes())}")
        return 0
    except ReleaseBundleError as error:
        print(
            json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
