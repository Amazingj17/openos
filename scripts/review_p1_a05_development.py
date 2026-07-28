from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
REPORT_ARTIFACTS = {
    "evaluation_report.json",
    "evaluation_per_slice.csv",
    "evaluation_per_seed.csv",
    "evaluation_primary_comparisons.csv",
}


class P1A05ReviewError(RuntimeError):
    pass


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    try:
        return _sha256(path.read_bytes())
    except OSError as error:
        raise P1A05ReviewError(f"cannot hash {path}: {error}") from error


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_constant
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise P1A05ReviewError(f"{label} is not strict UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise P1A05ReviewError(f"{label} must be a JSON object")
    return value


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise P1A05ReviewError(f"cannot canonicalize JSON: {error}") from error
    return _sha256(payload)


def _clean_head(repository: Path) -> str:
    try:
        commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        dirty = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise P1A05ReviewError(f"cannot inspect review commit: {error}") from error
    if dirty:
        raise P1A05ReviewError("independent review requires a clean tracked worktree")
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise P1A05ReviewError("Git did not return a full lowercase commit id")
    return commit


def _verify_remote_contains(repository: Path, commit: str, remote_ref: str) -> str:
    try:
        remote_commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", remote_ref],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "merge-base",
                "--is-ancestor",
                commit,
                remote_ref,
            ],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise P1A05ReviewError(
            f"cannot inspect immutable remote ref: {error}"
        ) from error
    if result.returncode != 0:
        raise P1A05ReviewError(
            f"candidate commit {commit} is not contained in {remote_ref}"
        )
    return remote_commit


def _comparison_candidate_commit(comparison: Mapping[str, Any], label: str) -> str:
    gates = comparison.get("gates", {})
    commit = comparison.get("code", {}).get("commit")
    required_gates = (
        "candidate_zero_failures_and_illegal_actions",
        "candidate_mean_ratio_below_heft_on_every_development_slice",
        "size_mean_ratio_below_one",
        "size_new_minus_p1_a04_ci_upper_below_zero",
        "id_mean_ratio_below_one_and_within_baseline_plus_0_02",
        "development_gate_passed",
    )
    if (
        comparison.get("task") != "P1-A05-DEVELOPMENT-COMPARISON"
        or comparison.get("decision") != "eligible_for_independent_review_before_G3"
        or any(gates.get(name) is not True for name in required_gates)
        or comparison.get("code", {}).get("working_tree_dirty") is not False
        or comparison.get("inputs", {}).get("test_accessed") is not False
        or comparison.get("inputs", {}).get("public_test") != "forbidden"
        or not isinstance(commit, str)
        or len(commit) != 40
        or any(char not in "0123456789abcdef" for char in commit)
    ):
        raise P1A05ReviewError(f"{label} comparison is not G3-review eligible")
    return commit


def _policy_seed_pairs(contract: Mapping[str, Any]) -> list[tuple[str, int]]:
    pairs: list[tuple[str, int]] = []
    for policy in contract.get("policies", []):
        if not isinstance(policy, dict) or not isinstance(
            policy.get("required_seeds"), list
        ):
            raise P1A05ReviewError("evaluation contract policy list is invalid")
        pairs.extend(
            (str(policy["id"]), int(seed)) for seed in policy["required_seeds"]
        )
    return pairs


def _development_slices(contract: Mapping[str, Any]) -> dict[str, int]:
    development = contract.get("modes", {}).get("development")
    slices = contract.get("slices")
    if not isinstance(development, list) or not isinstance(slices, list):
        raise P1A05ReviewError("evaluation contract development slices are invalid")
    counts = {
        str(item["id"]): int(item["scenario_count"])
        for item in slices
        if isinstance(item, dict) and item.get("id") in development
    }
    if set(counts) != set(development):
        raise P1A05ReviewError("evaluation contract slice inventory is incomplete")
    return counts


def _normalized_records(
    evidence: Mapping[str, Any],
    contract: Mapping[str, Any],
    contract_hash: str,
    *,
    candidate_commit: str,
    label: str,
) -> list[dict[str, Any]]:
    records = evidence.get("records")
    if (
        evidence.get("mode") != "development"
        or evidence.get("contract_sha256") != contract_hash
        or evidence.get("test_accessed") is not False
        or evidence.get("producer", {}).get("training_started") is not False
        or evidence.get("producer", {}).get("public_test_loaded") is not False
        or evidence.get("code", {}).get("commit") != candidate_commit
        or evidence.get("code", {}).get("working_tree_dirty") is not False
        or not isinstance(records, list)
        or evidence.get("records_sha256") != _canonical_sha256(records)
    ):
        raise P1A05ReviewError(f"{label} evidence boundary or records hash is invalid")
    slice_counts = _development_slices(contract)
    scenario_ids: dict[str, set[str]] = {slice_id: set() for slice_id in slice_counts}
    scenario_hashes: dict[tuple[str, str], set[str]] = {}
    observed: list[tuple[str, int, str, str]] = []
    normalized: list[dict[str, Any]] = []
    primary_policy = contract.get("primary_policy")
    for row in records:
        if not isinstance(row, dict):
            raise P1A05ReviewError(f"{label} evidence contains a non-object record")
        try:
            key = (
                str(row["policy"]),
                int(row["seed"]),
                str(row["slice_id"]),
                str(row["scenario_id"]),
            )
            runtime_ms = float(row["runtime_ms"])
            scenario_hash = str(row["scenario_hash"])
            score_ratio = float(row["score_ratio"])
        except (KeyError, TypeError, ValueError) as error:
            raise P1A05ReviewError(f"{label} evidence record is malformed") from error
        if (
            key[2] not in scenario_ids
            or not math.isfinite(runtime_ms)
            or runtime_ms < 0
            or not math.isfinite(score_ratio)
        ):
            raise P1A05ReviewError(f"{label} evidence record has invalid slice/runtime")
        if key[0] == primary_policy and (
            row.get("status") != "success"
            or row.get("penalty_applied") is not False
            or row.get("illegal_action_count") != 0
            or row.get("error_code") is not None
        ):
            raise P1A05ReviewError(
                f"{label} primary policy has a failure or illegal action"
            )
        scenario_ids[key[2]].add(key[3])
        scenario_hashes.setdefault((key[2], key[3]), set()).add(scenario_hash)
        observed.append(key)
        normalized.append(
            {name: value for name, value in row.items() if name != "runtime_ms"}
        )
    for slice_id, count in slice_counts.items():
        if len(scenario_ids[slice_id]) != count:
            raise P1A05ReviewError(f"{label} scenario count differs for {slice_id}")
    expected = {
        (policy, seed, slice_id, scenario_id)
        for policy, seed in _policy_seed_pairs(contract)
        for slice_id, ids in scenario_ids.items()
        for scenario_id in ids
    }
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise P1A05ReviewError(f"{label} evidence Cartesian product is incomplete")
    if any(len(hashes) != 1 for hashes in scenario_hashes.values()):
        raise P1A05ReviewError(f"{label} scenario hash differs across policies")
    return sorted(
        normalized,
        key=lambda row: (
            row["policy"],
            row["seed"],
            row["slice_id"],
            row["scenario_id"],
        ),
    )


def _verify_evidence_binding(
    comparison: Mapping[str, Any], evidence_path: Path, evidence: Mapping[str, Any]
) -> None:
    binding = comparison.get("inputs", {}).get("candidate_evidence", {})
    if binding.get("sha256") != _file_sha256(evidence_path) or binding.get(
        "records_sha256"
    ) != evidence.get("records_sha256"):
        raise P1A05ReviewError("comparison candidate evidence binding is invalid")


def _normalized_report(
    report_dir: Path,
    evidence_path: Path,
    evidence: Mapping[str, Any],
    comparison: Mapping[str, Any],
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    report_path = report_dir / "evaluation_report.json"
    manifest_path = report_dir / "evaluation_report_manifest.json"
    report = _load_json(report_path, f"{label} report")
    manifest = _load_json(manifest_path, f"{label} report manifest")
    artifacts = manifest.get("artifacts")
    if (
        report.get("report_scope") != "development"
        or report.get("evidence", {}).get("sha256") != _file_sha256(evidence_path)
        or report.get("evidence", {}).get("records_sha256")
        != evidence.get("records_sha256")
        or report.get("gate")
        != {
            "primary_zero_failures_and_illegal_actions": True,
            "primary_mean_ratio_below_reference_on_every_reported_slice": True,
            "release_publishable": False,
        }
        or manifest.get("test_accessed") is not False
        or not isinstance(artifacts, dict)
        or set(artifacts) != REPORT_ARTIFACTS
    ):
        raise P1A05ReviewError(f"{label} report boundary or inventory is invalid")
    for name, item in artifacts.items():
        path = report_dir / name
        if (
            not path.is_file()
            or not isinstance(item, dict)
            or item.get("bytes") != path.stat().st_size
            or item.get("sha256") != _file_sha256(path)
        ):
            raise P1A05ReviewError(f"{label} report artifact changed: {name}")
    standard_binding = comparison.get("inputs", {}).get("candidate_standard_report", {})
    if standard_binding.get(
        "bytes"
    ) != report_path.stat().st_size or standard_binding.get("sha256") != _file_sha256(
        report_path
    ):
        raise P1A05ReviewError(f"{label} comparison report binding is invalid")
    normalized = copy.deepcopy(report)
    normalized["evidence"].pop("sha256", None)
    normalized["evidence"].pop("records_sha256", None)
    for slice_report in normalized.get("slices", []):
        for policy_report in slice_report.get("policies", {}).values():
            policy_report.pop("runtime_ms", None)
    normalized_csvs: dict[str, Any] = {}
    for name in sorted(REPORT_ARTIFACTS - {"evaluation_report.json"}):
        try:
            with (report_dir / name).open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None or len(reader.fieldnames) != len(
                    set(reader.fieldnames)
                ):
                    raise P1A05ReviewError(f"{label} CSV header is invalid: {name}")
                retained = [
                    field
                    for field in reader.fieldnames
                    if not field.startswith("runtime_")
                ]
                normalized_csvs[name] = {
                    "columns": retained,
                    "rows": [
                        {field: row[field] for field in retained} for row in reader
                    ],
                }
        except (OSError, UnicodeError, csv.Error) as error:
            raise P1A05ReviewError(
                f"cannot normalize {label} CSV {name}: {error}"
            ) from error
    return (
        normalized,
        normalized_csvs,
        {
            "path": str(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "sha256": _file_sha256(manifest_path),
            "artifact_count": len(artifacts),
        },
    )


def _normalized_comparison(comparison: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(comparison)
    candidate = normalized["inputs"]["candidate_evidence"]
    candidate.pop("sha256", None)
    candidate.pop("records_sha256", None)
    report = normalized["inputs"]["candidate_standard_report"]
    report.pop("bytes", None)
    report.pop("sha256", None)
    return normalized


def _verify_checkpoints(
    repository: Path,
    formal: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> dict[str, Any]:
    formal_training = formal.get("inputs", {}).get("formal_training")
    replay_training = replay.get("inputs", {}).get("formal_training")
    if formal_training != replay_training or not isinstance(formal_training, dict):
        raise P1A05ReviewError("formal training binding differs in review replay")
    checkpoints = formal_training.get("checkpoints")
    if not isinstance(checkpoints, dict) or len(checkpoints) != 5:
        raise P1A05ReviewError("independent review requires exactly five checkpoints")
    verified: dict[str, Any] = {}
    for seed, item in checkpoints.items():
        path_value = item.get("path") if isinstance(item, dict) else None
        if not isinstance(path_value, str):
            raise P1A05ReviewError(f"checkpoint path is absent for seed {seed}")
        path = (repository / path_value).resolve()
        try:
            path.relative_to(repository.resolve())
        except ValueError as error:
            raise P1A05ReviewError(f"checkpoint escapes repository: {path}") from error
        if (
            not path.is_file()
            or item.get("bytes") != path.stat().st_size
            or item.get("sha256") != _file_sha256(path)
            or not isinstance(item.get("parameter_sha256"), str)
        ):
            raise P1A05ReviewError(f"checkpoint binding changed for seed {seed}")
        verified[str(seed)] = {
            "path": path_value,
            "bytes": item["bytes"],
            "sha256": item["sha256"],
            "parameter_sha256": item["parameter_sha256"],
        }
    return verified


def build_independent_review(
    *,
    repository: Path,
    remote_ref: str,
    contract_path: Path,
    formal_evidence_path: Path,
    formal_report_dir: Path,
    formal_comparison_path: Path,
    replay_evidence_path: Path,
    replay_report_dir: Path,
    replay_comparison_path: Path,
    output_path: Path,
) -> Path:
    root = repository.resolve()
    head = _clean_head(root)
    paths = (
        contract_path,
        formal_evidence_path,
        formal_report_dir / "evaluation_report.json",
        formal_comparison_path,
        replay_evidence_path,
        replay_report_dir / "evaluation_report.json",
        replay_comparison_path,
    )
    if any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise P1A05ReviewError(f"independent review inputs are absent: {missing}")
    if output_path.exists():
        raise P1A05ReviewError(f"refusing to overwrite review receipt: {output_path}")

    contract = _load_json(contract_path, "evaluation contract")
    contract_hash = _canonical_sha256(contract)
    formal_comparison = _load_json(formal_comparison_path, "formal comparison")
    replay_comparison = _load_json(replay_comparison_path, "review replay comparison")
    formal_commit = _comparison_candidate_commit(formal_comparison, "formal")
    replay_commit = _comparison_candidate_commit(replay_comparison, "review replay")
    if formal_commit != replay_commit or formal_commit != head:
        raise P1A05ReviewError(
            "formal/replay comparison and clean review HEAD must use one candidate commit"
        )
    remote_commit = _verify_remote_contains(root, head, remote_ref)

    formal_evidence = _load_json(formal_evidence_path, "formal evidence")
    replay_evidence = _load_json(replay_evidence_path, "review replay evidence")
    _verify_evidence_binding(formal_comparison, formal_evidence_path, formal_evidence)
    _verify_evidence_binding(replay_comparison, replay_evidence_path, replay_evidence)
    formal_records = _normalized_records(
        formal_evidence,
        contract,
        contract_hash,
        candidate_commit=head,
        label="formal",
    )
    replay_records = _normalized_records(
        replay_evidence,
        contract,
        contract_hash,
        candidate_commit=head,
        label="review replay",
    )
    if formal_records != replay_records:
        raise P1A05ReviewError(
            "review replay scheduling records differ after removing runtime_ms"
        )

    formal_report, formal_csvs, formal_report_manifest = _normalized_report(
        formal_report_dir,
        formal_evidence_path,
        formal_evidence,
        formal_comparison,
        label="formal",
    )
    replay_report, replay_csvs, replay_report_manifest = _normalized_report(
        replay_report_dir,
        replay_evidence_path,
        replay_evidence,
        replay_comparison,
        label="review replay",
    )
    if formal_report != replay_report:
        raise P1A05ReviewError(
            "review replay report differs after removing wall-clock/hash fields"
        )
    if formal_csvs != replay_csvs:
        raise P1A05ReviewError(
            "review replay CSVs differ after removing wall-clock columns"
        )
    formal_normalized_comparison = _normalized_comparison(formal_comparison)
    replay_normalized_comparison = _normalized_comparison(replay_comparison)
    if formal_normalized_comparison != replay_normalized_comparison:
        raise P1A05ReviewError(
            "review replay paired comparison differs after evidence/report hash removal"
        )
    checkpoints = _verify_checkpoints(root, formal_comparison, replay_comparison)
    script_path = Path(__file__).resolve()
    script_bytes = script_path.read_bytes()
    try:
        script_name = script_path.relative_to(root).as_posix()
    except ValueError:
        script_name = script_path.name
    try:
        normalized_script = (
            script_bytes.decode("utf-8")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .encode("utf-8")
        )
    except UnicodeError as error:
        raise P1A05ReviewError("review script is not strict UTF-8") from error

    receipt = {
        "format_version": 1,
        "task": "P1-A05-INDEPENDENT-DEVELOPMENT-REVIEW",
        "reviewer": "B",
        "candidate_commit": head,
        "code": {
            "commit": head,
            "working_tree_dirty": False,
            "script": {
                "path": script_name,
                "raw_sha256": _sha256(script_bytes),
                "normalized_lf_sha256": _sha256(normalized_script),
            },
        },
        "immutable_remote": {
            "ref": remote_ref,
            "ref_commit": remote_commit,
            "contains_candidate_commit": True,
        },
        "inputs": {
            "evaluation_contract": {
                "path": contract_path.name,
                "canonical_sha256": contract_hash,
            },
            "formal": {
                "evidence_sha256": _file_sha256(formal_evidence_path),
                "records_sha256": formal_evidence["records_sha256"],
                "report_manifest": formal_report_manifest,
                "comparison_sha256": _file_sha256(formal_comparison_path),
            },
            "review_replay": {
                "evidence_sha256": _file_sha256(replay_evidence_path),
                "records_sha256": replay_evidence["records_sha256"],
                "report_manifest": replay_report_manifest,
                "comparison_sha256": _file_sha256(replay_comparison_path),
            },
        },
        "normalized_equivalence": {
            "wall_clock_fields_removed": [
                "records[*].runtime_ms",
                "report.slices[*].policies[*].runtime_ms",
                "evidence/report file and records hashes",
            ],
            "record_count": len(formal_records),
            "records_canonical_sha256": _canonical_sha256(formal_records),
            "report_canonical_sha256": _canonical_sha256(formal_report),
            "csvs_canonical_sha256": _canonical_sha256(formal_csvs),
            "comparison_canonical_sha256": _canonical_sha256(
                formal_normalized_comparison
            ),
        },
        "checkpoints": checkpoints,
        "assertions": {
            "immutable_remote_commit_verified": True,
            "five_checkpoint_hashes_recomputed": True,
            "normalized_scheduling_records_equal": True,
            "normalized_reports_equal": True,
            "normalized_csvs_equal": True,
            "paired_comparisons_equal": True,
            "public_test_accessed": False,
        },
        "decision": "approve_before_g3",
        "claim_boundary": (
            "independent development replay only; never authorizes public-test access"
        ),
    }
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except OSError as error:
        output_path.unlink(missing_ok=True)
        raise P1A05ReviewError(f"cannot write review receipt: {error}") from error
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bind B's immutable P1-A05 development replay before G3"
    )
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--remote-ref", default="origin/main")
    parser.add_argument(
        "--contract",
        type=Path,
        default=REPOSITORY / "configs" / "p1_b02_evaluation_contract.json",
    )
    parser.add_argument(
        "--formal-evidence",
        type=Path,
        default=REPOSITORY
        / "outputs"
        / "p1-a05-development-evidence"
        / "development-evidence.json",
    )
    parser.add_argument(
        "--formal-report-dir",
        type=Path,
        default=REPOSITORY / "outputs" / "p1-a05-development-report",
    )
    parser.add_argument(
        "--formal-comparison",
        type=Path,
        default=REPOSITORY
        / "outputs"
        / "p1-a05-development-comparison"
        / "comparison.json",
    )
    parser.add_argument(
        "--replay-evidence",
        type=Path,
        default=REPOSITORY
        / "outputs"
        / "p1-a05-independent-review"
        / "development-evidence.json",
    )
    parser.add_argument(
        "--replay-report-dir",
        type=Path,
        default=REPOSITORY
        / "outputs"
        / "p1-a05-independent-review"
        / "development-report",
    )
    parser.add_argument(
        "--replay-comparison",
        type=Path,
        default=REPOSITORY
        / "outputs"
        / "p1-a05-independent-review"
        / "comparison"
        / "comparison.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY
        / "outputs"
        / "p1-a05-independent-review"
        / "p1_a05_independent_review.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_independent_review(
            repository=args.repository.resolve(),
            remote_ref=args.remote_ref,
            contract_path=args.contract.resolve(),
            formal_evidence_path=args.formal_evidence.resolve(),
            formal_report_dir=args.formal_report_dir.resolve(),
            formal_comparison_path=args.formal_comparison.resolve(),
            replay_evidence_path=args.replay_evidence.resolve(),
            replay_report_dir=args.replay_report_dir.resolve(),
            replay_comparison_path=args.replay_comparison.resolve(),
            output_path=args.output.resolve(),
        )
        print(result)
        return 0
    except P1A05ReviewError as error:
        print(
            json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
