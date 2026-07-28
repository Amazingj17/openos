from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from trisched.hashing import (
    canonical_json_sha256,
    file_sha256,
    normalized_lf_sha256,
    portable_text_hashes,
)


ROOT = Path(__file__).resolve().parents[1]


def test_normalized_hash_is_identical_across_lf_crlf_and_cr(
    tmp_path: Path,
) -> None:
    payloads = {
        "lf.py": b"alpha = 1\nbeta = 2\n",
        "crlf.py": b"alpha = 1\r\nbeta = 2\r\n",
        "cr.py": b"alpha = 1\rbeta = 2\r",
    }
    paths = []
    for name, payload in payloads.items():
        path = tmp_path / name
        path.write_bytes(payload)
        paths.append(path)

    assert len({file_sha256(path) for path in paths}) == 3
    assert len({normalized_lf_sha256(path) for path in paths}) == 1
    assert (
        normalized_lf_sha256(paths[0]) == hashlib.sha256(payloads["lf.py"]).hexdigest()
    )


def test_portable_text_hashes_keep_raw_execution_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.md"
    source.write_bytes("第一行\r\n第二行\r\n".encode("utf-8"))
    metadata = portable_text_hashes(source)
    assert metadata["bytes"] == source.stat().st_size
    assert metadata["raw_sha256"] == file_sha256(source)
    assert (
        metadata["normalized_lf_sha256"]
        == hashlib.sha256("第一行\n第二行\n".encode("utf-8")).hexdigest()
    )
    assert metadata["encoding"] == "utf-8"


def test_normalized_hash_rejects_non_utf8_bytes(tmp_path: Path) -> None:
    source = tmp_path / "invalid.py"
    source.write_bytes(b"\xff\xfe")
    with pytest.raises(UnicodeDecodeError):
        normalized_lf_sha256(source)


def test_canonical_json_hash_is_key_order_and_whitespace_independent() -> None:
    left = {"z": [1, 2], "name": "调度"}
    right = {"name": "调度", "z": [1, 2]}
    assert canonical_json_sha256(left) == canonical_json_sha256(right)


def test_gitattributes_freezes_portable_text_and_binary_contract() -> None:
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    for pattern in ("*.py", "*.json", "*.md", "*.yml", "*.ps1"):
        assert f"{pattern} text eol=lf" in attributes
    for pattern in ("*.npz", "*.zip", "*.docx", "*.pdf"):
        assert f"{pattern} binary" in attributes
