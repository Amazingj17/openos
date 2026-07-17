from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from trisched.benchmark import build_stg_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the reviewed STG split manifest for an exact source archive"
        )
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY / "data" / "benchmarks" / "stg-rnc50-hetero-v1.json",
    )
    args = parser.parse_args()
    manifest = build_stg_manifest(args.source)
    encoded = (
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(f"manifest: {args.output.resolve()}")
    print(f"sha256: {hashlib.sha256(encoded).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
