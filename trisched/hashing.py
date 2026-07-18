from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def file_sha256(path: str | Path) -> str:
    """Hash the exact bytes that were consumed or produced."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_lf_bytes(path: str | Path) -> bytes:
    """Return strict UTF-8 text with CRLF/CR normalized to LF."""

    text = Path(path).read_text(encoding="utf-8")
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def normalized_lf_sha256(path: str | Path) -> str:
    """Hash text semantics independently of a checkout's line endings."""

    return hashlib.sha256(normalized_lf_bytes(path)).hexdigest()


def portable_text_hashes(path: str | Path) -> dict[str, Any]:
    """Record both execution bytes and the cross-worktree text identity."""

    source = Path(path)
    return {
        "bytes": source.stat().st_size,
        "raw_sha256": file_sha256(source),
        "normalized_lf_sha256": normalized_lf_sha256(source),
        "encoding": "utf-8",
        "normalization": "CRLF/CR to LF; no other transformation",
    }


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
