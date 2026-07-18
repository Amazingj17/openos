from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def _markdown_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*.md")
        if not ({".git", ".pytest_cache", "outputs"} & set(path.parts))
    )


def test_local_markdown_links_resolve() -> None:
    missing: list[str] = []
    for source in _markdown_files():
        text = source.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.strip()
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1]
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            target = unquote(target.split("#", 1)[0])
            if not target:
                continue
            resolved = (
                ROOT / target.lstrip("/")
                if target.startswith("/")
                else source.parent / target
            ).resolve()
            if not resolved.exists():
                missing.append(f"{source.relative_to(ROOT).as_posix()} -> {target}")
    assert not missing, "missing local Markdown links:\n" + "\n".join(missing)


def test_markdown_fences_are_balanced() -> None:
    unbalanced = []
    for source in _markdown_files():
        fence_count = sum(
            line.lstrip().startswith("```")
            for line in source.read_text(encoding="utf-8").splitlines()
        )
        if fence_count % 2:
            unbalanced.append(source.relative_to(ROOT).as_posix())
    assert not unbalanced, "unbalanced Markdown fences:\n" + "\n".join(unbalanced)
