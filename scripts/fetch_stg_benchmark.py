from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from trisched.benchmark import (
    SOURCE_ARCHIVE_BYTES,
    SOURCE_ARCHIVE_NAME,
    SOURCE_ARCHIVE_SHA256,
    SOURCE_ARCHIVE_URL,
    extract_verified_archive,
    load_benchmark_manifest,
    verify_archive,
    verify_frozen_splits,
)


def download_archive(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    request = urllib.request.Request(
        SOURCE_ARCHIVE_URL,
        headers={"User-Agent": "TriSched-P1-B01/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with temporary.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
        verify_archive(temporary)
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch and verify the pinned CC-BY STG rnc50 JSON archive"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY / "data" / "benchmarks" / "stg-rnc50-hetero-v1.json",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPOSITORY / "outputs" / "benchmarks" / "stg-rnc50-hetero-v1",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="refuse network access; the exact archive must already be cached",
    )
    args = parser.parse_args(argv)

    manifest = load_benchmark_manifest(args.manifest)
    archive_path = args.cache_dir / SOURCE_ARCHIVE_NAME
    if archive_path.exists():
        verify_archive(archive_path)
    elif args.offline:
        raise FileNotFoundError(f"offline archive is missing: {archive_path}")
    else:
        download_archive(archive_path)

    raw_root = args.cache_dir / "raw"
    if not raw_root.exists():
        expected_files = [entry["source"] for entry in manifest["entries"]]
        extract_verified_archive(archive_path, raw_root, expected_files)
    splits = verify_frozen_splits(raw_root, args.manifest)
    result = {
        "ok": True,
        "archive": {
            "path": str(archive_path.resolve()),
            "bytes": SOURCE_ARCHIVE_BYTES,
            "sha256": SOURCE_ARCHIVE_SHA256,
        },
        "raw_root": str(raw_root.resolve()),
        "splits": {
            name: {
                "count": len(values),
                "scenario_hashes_sha256": manifest["splits"][name][
                    "scenario_hashes_sha256"
                ],
            }
            for name, values in splits.items()
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
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
        raise SystemExit(2)
