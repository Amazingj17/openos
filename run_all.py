"""One-command training entry point for TriSched.

This thin wrapper keeps the full training workflow in
``scripts.run_model_comparison`` while giving users a memorable command:

    python run_all.py
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
import sys

from scripts.fetch_stg_benchmark import main as _fetch_stg_benchmark
from scripts.run_model_comparison import main as _run_model_comparison
from trisched.visualization import write_model_comparison_visualizations

REPOSITORY = Path(__file__).resolve().parent
DEFAULT_BENCHMARK_ROOT = (
    REPOSITORY / "outputs" / "benchmarks" / "stg-rnc50-hetero-v1" / "raw"
)
DEFAULT_COMPARISON_ROOT = REPOSITORY / "outputs" / "stg-model-comparison"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reuse_completed_default_results() -> Path | None:
    """Validate and reuse the complete frozen comparison for a quick demo."""

    results = DEFAULT_COMPARISON_ROOT / "results"
    manifest_path = results / "comparison_manifest.json"
    summary_path = DEFAULT_COMPARISON_ROOT / "comparison_pipeline_summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    required = (
        "comparison.json",
        "comparison_per_instance.csv",
        "comparison_per_seed.csv",
        "comparison_per_scenario.csv",
    )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or summary.get("test_accessed") is not False:
        return None
    for name in required:
        metadata = artifacts.get(name)
        artifact = results / name
        if (
            not isinstance(metadata, dict)
            or not artifact.is_file()
            or artifact.stat().st_size != metadata.get("bytes")
            or _sha256(artifact) != metadata.get("sha256")
        ):
            return None

    comparison = json.loads((results / "comparison.json").read_text(encoding="utf-8"))
    paper_aligned = results / "paper-aligned"
    html_path, _ = write_model_comparison_visualizations(
        comparison,
        html_path=paper_aligned / "comparison.html",
        svg_path=paper_aligned / "comparison.svg",
        json_name="../comparison.json",
        per_instance_name="../comparison_per_instance.csv",
        per_seed_name="../comparison_per_seed.csv",
        per_scenario_name="../comparison_per_scenario.csv",
    )
    models = summary["models"]
    paired = summary["paired_validation"]
    interval = paired["hierarchical_paired_bootstrap"]
    print("[demo] verified and reused the complete frozen validation evidence")
    print(
        "[demo] Masked MLP mean ratio="
        f"{models['masked_mlp']['mean_ratio']:.6f}"
    )
    print(
        "[demo] TriSched-GNN-PPO mean ratio="
        f"{models['task_gnn']['mean_ratio']:.6f}"
    )
    print(
        "[demo] paired 95% CI="
        f"[{interval['lower']:.6f}, {interval['upper']:.6f}]"
    )
    print(f"[demo] report: {html_path.resolve()}")
    print(
        "[demo] use --output outputs/<new-directory> to force a fresh "
        "end-to-end run"
    )
    return html_path


def _prepare_benchmark() -> int:
    """Download and verify the pinned benchmark when it is not cached."""

    if DEFAULT_BENCHMARK_ROOT.is_dir():
        return 0
    print("[0/4] STG training data is missing; downloading and verifying it")
    return _fetch_stg_benchmark([])


def main(argv: Sequence[str] | None = None) -> int:
    """Train all supported comparison models and generate the report."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    help_requested = any(
        value in {"-h", "--help"} for value in arguments
    )
    if not arguments and _reuse_completed_default_results() is not None:
        return 0
    if not help_requested and _prepare_benchmark() != 0:
        return 2
    return _run_model_comparison(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
