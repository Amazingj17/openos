from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "MODEL_RESULT_MANIFEST.json"
_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
_PRIVATE_KEY_MARKERS = (
    b"-----BEGIN " + b"PRIVATE KEY-----",
    b"-----BEGIN " + b"OPENSSH PRIVATE KEY-----",
    b"-----BEGIN " + b"RSA PRIVATE KEY-----",
)


class P106BundleError(RuntimeError):
    pass


@dataclass(frozen=True)
class Artifact:
    archive_path: str
    source_path: Path
    role: str


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256(path.read_bytes())


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _load_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"), parse_constant=_reject_json_constant
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise P106BundleError(f"{label} is not strict UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise P106BundleError(f"{label} must be a JSON object")
    return value


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return _load_json_bytes(path.read_bytes(), label)
    except OSError as error:
        raise P106BundleError(f"{label} cannot be read: {error}") from error


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise P106BundleError(f"cannot canonicalize JSON: {error}") from error
    return _sha256(payload)


def _write_json_new(path: Path, value: Mapping[str, Any]) -> Path:
    if path.exists():
        raise P106BundleError(f"refusing to overwrite existing file: {path}")
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    except OSError as error:
        path.unlink(missing_ok=True)
        raise P106BundleError(f"cannot write {path}: {error}") from error
    return path


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
        raise P106BundleError(f"cannot inspect release commit: {error}") from error
    if dirty:
        raise P106BundleError("P1-06 requires a clean tracked worktree")
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise P106BundleError("Git did not return a full lowercase commit id")
    return commit


def _repository_file(repository: Path, path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(repository.resolve())
    except ValueError as error:
        raise P106BundleError(f"{label} escapes the repository: {path}") from error
    if not resolved.is_file():
        raise P106BundleError(f"{label} does not exist: {resolved}")
    return resolved


def _safe_archive_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise P106BundleError(f"unsafe model bundle path: {value!r}")
    if "public_test" in value.lower() or "public-test" in value.lower():
        raise P106BundleError(f"public-test path is forbidden in model bundle: {value}")
    return value


def _artifact_record(repository: Path, artifact: Artifact) -> dict[str, Any]:
    archive_path = _safe_archive_path(artifact.archive_path)
    source = _repository_file(repository, artifact.source_path, artifact.role)
    payload = source.read_bytes()
    if any(marker in payload for marker in _PRIVATE_KEY_MARKERS):
        raise P106BundleError(f"private-key marker in {artifact.role}: {source}")
    return {
        "path": archive_path,
        "source": source.relative_to(repository.resolve()).as_posix(),
        "role": artifact.role,
        "bytes": len(payload),
        "sha256": _sha256(payload),
    }


def _expect_record(path: Path, value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise P106BundleError(f"{label} binding is absent")
    if value.get("bytes") != path.stat().st_size or value.get("sha256") != _file_sha256(
        path
    ):
        raise P106BundleError(f"{label} bytes or SHA-256 changed")


def _validate_contract(contract: Mapping[str, Any]) -> None:
    independent_review = contract.get("p1_a05_independent_review")
    p1_06_review = contract.get("p1_06_review")
    if (
        contract.get("format_version") != 1
        or contract.get("task_id") != "P1-06"
        or contract.get("bundle_type") != "trisched_model_results_release"
        or contract.get("expected_seeds")
        != [20260717, 20260718, 20260719, 20260720, 20260721]
        or contract.get("comparison", {}).get("task") != "P1-A05-DEVELOPMENT-COMPARISON"
        or contract.get("comparison", {}).get("required_decision")
        != "eligible_for_independent_review_before_G3"
        or contract.get("development_report_artifacts")
        != [
            "evaluation_per_seed.csv",
            "evaluation_per_slice.csv",
            "evaluation_primary_comparisons.csv",
            "evaluation_report.json",
            "evaluation_report_manifest.json",
        ]
        or contract.get("g3")
        != {
            "task": "G3",
            "required_decision": "approve_p1_a05_as_primary",
            "members": ["A", "B"],
        }
        or independent_review
        != {
            "task": "P1-A05-INDEPENDENT-DEVELOPMENT-REVIEW",
            "reviewer": "B",
            "required_decision": "approve_before_g3",
            "required_assertions": [
                "immutable_remote_commit_verified",
                "five_checkpoint_hashes_recomputed",
                "normalized_scheduling_records_equal",
                "normalized_reports_equal",
                "normalized_csvs_equal",
                "paired_comparisons_equal",
            ],
            "public_test_accessed": False,
        }
        or p1_06_review
        != {
            "task": "P1-06",
            "members": ["A", "B"],
            "required_assertions": [
                "checkpoint_hashes_recomputed",
                "development_result_hashes_recomputed",
                "minimal_ablation_reviewed",
                "package_inventory_reviewed",
            ],
        }
        or contract.get("public_test")
        != {
            "accessed": False,
            "status": "forbidden",
            "raw_bytes_in_bundle": False,
        }
    ):
        raise P106BundleError("P1-06 bundle contract has unsupported semantics")


def _validate_comparison(
    comparison: Mapping[str, Any], contract: Mapping[str, Any]
) -> str:
    expected = contract["comparison"]
    gates = comparison.get("gates", {})
    inputs = comparison.get("inputs", {})
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
        comparison.get("task") != expected["task"]
        or comparison.get("decision") != expected["required_decision"]
        or any(gates.get(name) is not True for name in required_gates)
        or comparison.get("code", {}).get("working_tree_dirty") is not False
        or inputs.get("test_accessed") is not False
        or inputs.get("public_test") != "forbidden"
        or not isinstance(commit, str)
        or len(commit) != 40
        or any(char not in "0123456789abcdef" for char in commit)
    ):
        raise P106BundleError("comparison does not authorize G3 review")
    return commit


def _validate_independent_review(
    review: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    candidate_commit: str,
    comparison_sha256: str,
    expected_checkpoints: Mapping[str, Any],
) -> None:
    expected = contract["p1_a05_independent_review"]
    assertions = review.get("assertions")
    script = review.get("code", {}).get("script")
    normalized = review.get("normalized_equivalence")
    if (
        review.get("format_version") != 1
        or review.get("task") != expected["task"]
        or review.get("reviewer") != expected["reviewer"]
        or review.get("decision") != expected["required_decision"]
        or review.get("candidate_commit") != candidate_commit
        or review.get("code", {}).get("commit") != candidate_commit
        or review.get("code", {}).get("working_tree_dirty") is not False
        or not isinstance(script, dict)
        or not isinstance(script.get("path"), str)
        or any(
            not isinstance(script.get(name), str) or len(script[name]) != 64
            for name in ("raw_sha256", "normalized_lf_sha256")
        )
        or review.get("immutable_remote", {}).get("contains_candidate_commit")
        is not True
        or review.get("inputs", {}).get("formal", {}).get("comparison_sha256")
        != comparison_sha256
        or review.get("checkpoints") != expected_checkpoints
        or not isinstance(normalized, dict)
        or not isinstance(normalized.get("record_count"), int)
        or normalized["record_count"] <= 0
        or any(
            not isinstance(normalized.get(name), str) or len(normalized[name]) != 64
            for name in (
                "records_canonical_sha256",
                "report_canonical_sha256",
                "csvs_canonical_sha256",
                "comparison_canonical_sha256",
            )
        )
        or not isinstance(assertions, dict)
        or any(
            assertions.get(name) is not True for name in expected["required_assertions"]
        )
        or assertions.get("public_test_accessed") is not False
    ):
        raise P106BundleError(
            "P1-A05 independent review does not bind the immutable candidate"
        )


def _validate_g3_authorization(
    authorization: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    candidate_commit: str,
    release_commit: str,
    comparison_sha256: str,
    independent_review_sha256: str,
) -> None:
    expected = contract["g3"]
    if (
        authorization.get("format_version") != 1
        or authorization.get("task") != expected["task"]
        or authorization.get("decision") != expected["required_decision"]
        or authorization.get("candidate_commit") != candidate_commit
        or authorization.get("release_commit") != release_commit
        or authorization.get("comparison_sha256") != comparison_sha256
        or authorization.get("independent_review_sha256") != independent_review_sha256
        or authorization.get("development_gate_passed") is not True
        or authorization.get("test_accessed") is not False
        or authorization.get("public_test") != "forbidden"
    ):
        raise P106BundleError("G3 authorization does not bind the approved candidate")
    approvals = authorization.get("approvals")
    if not isinstance(approvals, list) or len(approvals) != 2:
        raise P106BundleError("G3 authorization requires exactly two approvals")
    by_member = {
        item.get("member"): item
        for item in approvals
        if isinstance(item, dict) and isinstance(item.get("member"), str)
    }
    if set(by_member) != set(expected["members"]) or any(
        by_member[member].get("decision") != "approve" for member in expected["members"]
    ):
        raise P106BundleError("G3 authorization must be approved by A and B")


def _report_artifacts(
    repository: Path,
    contract: Mapping[str, Any],
    comparison: Mapping[str, Any],
    report_dir: Path,
) -> list[Artifact]:
    required = list(contract["development_report_artifacts"])
    manifest_path = _repository_file(
        repository,
        report_dir / "evaluation_report_manifest.json",
        "development report manifest",
    )
    manifest = _load_json(manifest_path, "development report manifest")
    if manifest.get("test_accessed") is not False:
        raise P106BundleError("development report manifest accessed public test")
    listed = manifest.get("artifacts")
    if not isinstance(listed, dict) or set(listed) != set(required) - {
        "evaluation_report_manifest.json"
    }:
        raise P106BundleError("development report artifact inventory changed")
    artifacts: list[Artifact] = []
    for name in required:
        path = _repository_file(
            repository, report_dir / name, f"report artifact {name}"
        )
        if name != "evaluation_report_manifest.json":
            _expect_record(path, listed.get(name), f"report artifact {name}")
        artifacts.append(
            Artifact(f"results/development-report/{name}", path, "development_report")
        )
    report_path = report_dir / "evaluation_report.json"
    report = _load_json(report_path, "development report")
    binding = comparison.get("inputs", {}).get("candidate_standard_report")
    _expect_record(report_path, binding, "comparison development report")
    if (
        report.get("report_scope") != "development"
        or report.get("gate", {}).get("primary_zero_failures_and_illegal_actions")
        is not True
        or report.get("gate", {}).get(
            "primary_mean_ratio_below_reference_on_every_reported_slice"
        )
        is not True
        or report.get("gate", {}).get("release_publishable") is not False
    ):
        raise P106BundleError("standard development report gates are inconsistent")
    return artifacts


def _collect_review_inputs(
    *,
    repository: Path,
    contract_path: Path,
    evaluation_contract_path: Path,
    preregister_path: Path,
    implementation_review_path: Path,
    comparison_path: Path,
    baseline_evidence_path: Path,
    candidate_evidence_path: Path,
    candidate_report_dir: Path,
    training_dir: Path,
    independent_review_path: Path,
    g3_authorization_path: Path,
) -> tuple[dict[str, Any], list[Artifact]]:
    root = repository.resolve()
    release_commit = _clean_head(root)
    contract_path = _repository_file(root, contract_path, "P1-06 contract")
    contract = _load_json(contract_path, "P1-06 contract")
    _validate_contract(contract)
    comparison_path = _repository_file(root, comparison_path, "comparison")
    comparison = _load_json(comparison_path, "comparison")
    candidate_commit = _validate_comparison(comparison, contract)
    comparison_sha256 = _file_sha256(comparison_path)
    inputs = comparison["inputs"]

    evaluation_contract_path = _repository_file(
        root, evaluation_contract_path, "evaluation contract"
    )
    evaluation_contract = _load_json(evaluation_contract_path, "evaluation contract")
    if inputs.get("contract", {}).get("canonical_sha256") != _canonical_json_sha256(
        evaluation_contract
    ):
        raise P106BundleError("evaluation contract differs from comparison")

    preregister_path = _repository_file(root, preregister_path, "preregister")
    preregister = _load_json(preregister_path, "preregister")
    if inputs.get("preregister", {}).get("canonical_sha256") != _canonical_json_sha256(
        preregister
    ):
        raise P106BundleError("preregister differs from comparison")

    implementation_review_path = _repository_file(
        root, implementation_review_path, "implementation review"
    )
    formal = inputs.get("formal_training", {})
    if formal.get("implementation_review_sha256") != _file_sha256(
        implementation_review_path
    ):
        raise P106BundleError("implementation review differs from comparison")

    baseline_evidence_path = _repository_file(
        root, baseline_evidence_path, "baseline evidence"
    )
    candidate_evidence_path = _repository_file(
        root, candidate_evidence_path, "candidate evidence"
    )
    baseline = _load_json(baseline_evidence_path, "baseline evidence")
    candidate = _load_json(candidate_evidence_path, "candidate evidence")
    for label, path, evidence, binding in (
        (
            "baseline",
            baseline_evidence_path,
            baseline,
            inputs.get("baseline_evidence", {}),
        ),
        (
            "candidate",
            candidate_evidence_path,
            candidate,
            inputs.get("candidate_evidence", {}),
        ),
    ):
        if (
            binding.get("sha256") != _file_sha256(path)
            or binding.get("records_sha256") != evidence.get("records_sha256")
            or evidence.get("mode") != "development"
            or evidence.get("test_accessed") is not False
            or evidence.get("producer", {}).get("public_test_loaded") is not False
        ):
            raise P106BundleError(
                f"{label} evidence differs from comparison or boundary"
            )
    if (
        candidate.get("code", {}).get("commit") != candidate_commit
        or candidate.get("code", {}).get("working_tree_dirty") is not False
    ):
        raise P106BundleError(
            "candidate evidence is not from the clean candidate commit"
        )

    training_summary_path = _repository_file(
        root, training_dir / "p1_a05_summary.json", "training summary"
    )
    training_manifest_path = _repository_file(
        root, training_dir / "p1_a05_run_manifest.json", "training manifest"
    )
    _expect_record(training_summary_path, formal.get("summary"), "training summary")
    _expect_record(
        training_manifest_path, formal.get("run_manifest"), "training manifest"
    )
    summary = _load_json(training_summary_path, "training summary")
    manifest = _load_json(training_manifest_path, "training manifest")
    if (
        summary.get("mode") != "p1_a05_size_robustness"
        or summary.get("formal_run_count") != 1
        or summary.get("validation_gate_passed") is not True
        or summary.get("data_access", {}).get("test_accessed") is not False
        or summary.get("data_access", {}).get("public_test") != "forbidden"
        or manifest.get("code", {}).get("commit") != candidate_commit
        or manifest.get("code", {}).get("working_tree_dirty") is not False
        or manifest.get("execution", {}).get("formal_run_count") != 1
        or manifest.get("inputs", {}).get("test_accessed") is not False
        or manifest.get("inputs", {}).get("public_test") != "forbidden"
    ):
        raise P106BundleError("formal training artifacts do not satisfy release gates")

    checkpoint_bindings = formal.get("checkpoints")
    manifest_checkpoints = manifest.get("checkpoints")
    expected_seeds = [str(seed) for seed in contract["expected_seeds"]]
    if (
        not isinstance(checkpoint_bindings, dict)
        or list(checkpoint_bindings) != expected_seeds
        or not isinstance(manifest_checkpoints, dict)
        or set(manifest_checkpoints) != set(expected_seeds)
    ):
        raise P106BundleError(
            "final checkpoint seed inventory is not exactly five seeds"
        )
    checkpoint_artifacts: list[Artifact] = []
    for seed in expected_seeds:
        binding = checkpoint_bindings[seed]
        path_value = binding.get("path") if isinstance(binding, dict) else None
        if not isinstance(path_value, str):
            raise P106BundleError(f"checkpoint path is absent for seed {seed}")
        checkpoint_path = _repository_file(
            root, root / path_value, f"checkpoint {seed}"
        )
        _expect_record(checkpoint_path, binding, f"checkpoint {seed}")
        manifest_actor = manifest_checkpoints.get(seed, {}).get("best", {}).get("actor")
        if (
            not isinstance(manifest_actor, dict)
            or manifest_actor.get("name") != checkpoint_path.name
            or manifest_actor.get("sha256") != binding.get("sha256")
            or manifest_actor.get("parameter_sha256") != binding.get("parameter_sha256")
        ):
            raise P106BundleError(
                f"training manifest checkpoint differs for seed {seed}"
            )
        checkpoint_artifacts.append(
            Artifact(
                f"models/{checkpoint_path.name}", checkpoint_path, "primary_checkpoint"
            )
        )

    report_artifacts = _report_artifacts(
        root, contract, comparison, candidate_report_dir
    )
    independent_review_path = _repository_file(
        root, independent_review_path, "P1-A05 independent review"
    )
    independent_review = _load_json(
        independent_review_path, "P1-A05 independent review"
    )
    _validate_independent_review(
        independent_review,
        contract,
        candidate_commit=candidate_commit,
        comparison_sha256=comparison_sha256,
        expected_checkpoints=formal.get("checkpoints", {}),
    )
    independent_review_sha256 = _file_sha256(independent_review_path)
    g3_authorization_path = _repository_file(
        root, g3_authorization_path, "G3 authorization"
    )
    g3 = _load_json(g3_authorization_path, "G3 authorization")
    _validate_g3_authorization(
        g3,
        contract,
        candidate_commit=candidate_commit,
        release_commit=release_commit,
        comparison_sha256=comparison_sha256,
        independent_review_sha256=independent_review_sha256,
    )

    artifacts = [
        Artifact(
            "provenance/p1_06_bundle_contract.json", contract_path, "bundle_contract"
        ),
        Artifact(
            "provenance/p1_b02_evaluation_contract.json",
            evaluation_contract_path,
            "evaluation_contract",
        ),
        Artifact(
            "provenance/p1_a05_size_robustness_preregister.json",
            preregister_path,
            "preregister",
        ),
        Artifact(
            "provenance/p1_a05_implementation_review.json",
            implementation_review_path,
            "implementation_review",
        ),
        Artifact(
            "training/p1_a05_summary.json", training_summary_path, "training_summary"
        ),
        Artifact(
            "training/p1_a05_run_manifest.json",
            training_manifest_path,
            "training_manifest",
        ),
        *checkpoint_artifacts,
        Artifact(
            "results/baseline-p1-a04-development-evidence.json",
            baseline_evidence_path,
            "minimal_ablation_baseline",
        ),
        Artifact(
            "results/candidate-p1-a05-development-evidence.json",
            candidate_evidence_path,
            "primary_development_evidence",
        ),
        *report_artifacts,
        Artifact(
            "results/p1_a05_development_comparison.json",
            comparison_path,
            "minimal_ablation_comparison",
        ),
        Artifact(
            "governance/p1_a05_independent_review.json",
            independent_review_path,
            "p1_a05_independent_review",
        ),
        Artifact(
            "governance/g3_authorization.json",
            g3_authorization_path,
            "g3_authorization",
        ),
    ]
    records = sorted(
        (_artifact_record(root, artifact) for artifact in artifacts),
        key=lambda item: item["path"],
    )
    if len({item["path"] for item in records}) != len(records):
        raise P106BundleError("duplicate archive path in P1-06 inventory")
    request = {
        "format_version": 1,
        "mode": "p1_06_review_request",
        "task": "P1-06",
        "candidate_commit": candidate_commit,
        "release_commit": release_commit,
        "comparison_sha256": comparison_sha256,
        "independent_review_sha256": independent_review_sha256,
        "g3_authorization_sha256": _file_sha256(g3_authorization_path),
        "minimal_ablation": {
            "definition": contract["comparison"]["minimal_ablation"],
            "artifact": "results/p1_a05_development_comparison.json",
            "decision": comparison["decision"],
        },
        "gates": {
            "development_gate_passed": True,
            "independent_replay_approved_by_b": True,
            "g3_approved_by_a_and_b": True,
            "five_primary_checkpoints_bound": True,
        },
        "public_test": {
            "accessed": False,
            "status": "forbidden",
            "raw_bytes_in_bundle": False,
        },
        "package_artifacts": records,
    }
    return request, artifacts


def prepare_review_request(*, output_path: Path, **kwargs: Any) -> Path:
    request, _ = _collect_review_inputs(**kwargs)
    return _write_json_new(output_path.resolve(), request)


def _validate_review_receipts(
    receipts: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    request: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    expected = contract["p1_06_review"]
    request_sha256 = _canonical_json_sha256(request)
    by_member: dict[str, Mapping[str, Any]] = {}
    for receipt in receipts:
        member = receipt.get("member")
        assertions = receipt.get("assertions")
        if (
            receipt.get("format_version") != 1
            or receipt.get("task") != expected["task"]
            or member not in expected["members"]
            or receipt.get("decision") != "approve"
            or receipt.get("request_canonical_sha256") != request_sha256
            or receipt.get("candidate_commit") != request.get("candidate_commit")
            or receipt.get("release_commit") != request.get("release_commit")
            or receipt.get("public_test_accessed") is not False
            or not isinstance(assertions, dict)
            or any(
                assertions.get(name) is not True
                for name in expected["required_assertions"]
            )
        ):
            raise P106BundleError("P1-06 review receipt is incomplete or mismatched")
        if member in by_member:
            raise P106BundleError(f"duplicate P1-06 review member: {member}")
        by_member[str(member)] = receipt
    if set(by_member) != set(expected["members"]):
        raise P106BundleError("P1-06 requires independent receipts from A and B")
    return by_member


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=_FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def build_model_result_bundle(
    *,
    output_path: Path,
    review_request_path: Path,
    review_receipt_paths: Sequence[Path],
    **kwargs: Any,
) -> Path:
    repository = Path(kwargs["repository"]).resolve()
    request, artifacts = _collect_review_inputs(**kwargs)
    review_request_path = _repository_file(
        repository, review_request_path, "P1-06 review request"
    )
    stored_request = _load_json(review_request_path, "P1-06 review request")
    if stored_request != request:
        raise P106BundleError(
            "P1-06 review request no longer matches current artifacts"
        )
    contract_path = _repository_file(
        repository, Path(kwargs["contract_path"]), "P1-06 contract"
    )
    contract = _load_json(contract_path, "P1-06 contract")
    receipt_paths = [
        _repository_file(repository, path, "P1-06 review receipt")
        for path in review_receipt_paths
    ]
    receipt_objects = [
        _load_json(path, f"P1-06 review receipt {path.name}") for path in receipt_paths
    ]
    by_member = _validate_review_receipts(receipt_objects, contract, request)

    governance_artifacts = [
        Artifact(
            "governance/p1_06_review_request.json",
            review_request_path,
            "p1_06_review_request",
        )
    ]
    for member in contract["p1_06_review"]["members"]:
        index = receipt_objects.index(by_member[member])
        governance_artifacts.append(
            Artifact(
                f"governance/p1_06_review_{member.lower()}.json",
                receipt_paths[index],
                "p1_06_review_receipt",
            )
        )
    all_artifacts = [*artifacts, *governance_artifacts]
    records = sorted(
        (_artifact_record(repository, artifact) for artifact in all_artifacts),
        key=lambda item: item["path"],
    )
    manifest = {
        "format_version": 1,
        "bundle_type": contract["bundle_type"],
        "candidate_commit": request["candidate_commit"],
        "release_commit": request["release_commit"],
        "review_request_canonical_sha256": _canonical_json_sha256(request),
        "approvals": [
            {
                "member": member,
                "decision": "approve",
                "receipt_sha256": _file_sha256(
                    receipt_paths[receipt_objects.index(by_member[member])]
                ),
            }
            for member in contract["p1_06_review"]["members"]
        ],
        "archive_policy": {
            "entry_order": "UTF-8 path sort",
            "entry_timestamp": "1980-01-01T00:00:00",
            "public_test_accessed": False,
            "public_test_raw_bytes_included": False,
            "credentials_included": False,
        },
        "files": records,
    }
    manifest_payload = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    target = output_path.resolve()
    if target.exists():
        raise P106BundleError(f"refusing to overwrite model/result bundle: {target}")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        source_by_archive = {
            artifact.archive_path: artifact.source_path for artifact in all_artifacts
        }
        with zipfile.ZipFile(
            target,
            mode="x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for record in records:
                archive.writestr(
                    _zip_info(record["path"]),
                    source_by_archive[record["path"]].read_bytes(),
                )
            archive.writestr(_zip_info(MANIFEST_NAME), manifest_payload)
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as error:
        target.unlink(missing_ok=True)
        raise P106BundleError(f"cannot write model/result bundle: {error}") from error
    verify_model_result_bundle(target)
    return target


def verify_model_result_bundle(path: Path) -> dict[str, Any]:
    source = path.resolve()
    try:
        with zipfile.ZipFile(source, "r") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or MANIFEST_NAME not in names:
                raise P106BundleError("bundle has duplicate entries or no manifest")
            manifest = _load_json_bytes(archive.read(MANIFEST_NAME), "bundle manifest")
            records = manifest.get("files")
            if not isinstance(records, list):
                raise P106BundleError("bundle manifest file inventory is absent")
            expected_names = [item.get("path") for item in records]
            if expected_names != sorted(expected_names) or names != [
                *expected_names,
                MANIFEST_NAME,
            ]:
                raise P106BundleError("bundle entry order differs from manifest")
            for item in records:
                name = _safe_archive_path(item["path"])
                payload = archive.read(name)
                if len(payload) != item.get("bytes") or _sha256(payload) != item.get(
                    "sha256"
                ):
                    raise P106BundleError(f"bundle payload hash mismatch: {name}")
                if any(marker in payload for marker in _PRIVATE_KEY_MARKERS):
                    raise P106BundleError(f"private-key marker in bundle: {name}")
            if (
                manifest.get("bundle_type") != "trisched_model_results_release"
                or manifest.get("archive_policy", {}).get("public_test_accessed")
                is not False
                or manifest.get("archive_policy", {}).get(
                    "public_test_raw_bytes_included"
                )
                is not False
            ):
                raise P106BundleError("bundle manifest release boundary is invalid")
            request = _load_json_bytes(
                archive.read("governance/p1_06_review_request.json"),
                "bundled review request",
            )
            contract = _load_json_bytes(
                archive.read("provenance/p1_06_bundle_contract.json"),
                "bundled P1-06 contract",
            )
            _validate_contract(contract)
            receipts = [
                _load_json_bytes(
                    archive.read(f"governance/p1_06_review_{member.lower()}.json"),
                    f"bundled {member} receipt",
                )
                for member in contract["p1_06_review"]["members"]
            ]
            _validate_review_receipts(receipts, contract, request)
            if manifest.get(
                "review_request_canonical_sha256"
            ) != _canonical_json_sha256(request):
                raise P106BundleError("bundle manifest review request hash differs")
    except (OSError, KeyError, TypeError, zipfile.BadZipFile) as error:
        if isinstance(error, P106BundleError):
            raise
        raise P106BundleError(f"cannot verify model/result bundle: {error}") from error
    return manifest


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument(
        "--contract",
        type=Path,
        default=REPOSITORY / "configs" / "p1_06_bundle_contract.json",
    )
    parser.add_argument(
        "--evaluation-contract",
        type=Path,
        default=REPOSITORY / "configs" / "p1_b02_evaluation_contract.json",
    )
    parser.add_argument(
        "--preregister",
        type=Path,
        default=REPOSITORY / "configs" / "p1_a05_size_robustness_preregister.json",
    )
    parser.add_argument(
        "--implementation-review",
        type=Path,
        default=REPOSITORY / "configs" / "p1_a05_implementation_review.json",
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=REPOSITORY
        / "outputs"
        / "p1-a05-development-comparison"
        / "comparison.json",
    )
    parser.add_argument(
        "--baseline-evidence",
        type=Path,
        default=REPOSITORY
        / "outputs"
        / "p1-b02-development-evidence"
        / "development-evidence.json",
    )
    parser.add_argument(
        "--candidate-evidence",
        type=Path,
        default=REPOSITORY
        / "outputs"
        / "p1-a05-development-evidence"
        / "development-evidence.json",
    )
    parser.add_argument(
        "--candidate-report-dir",
        type=Path,
        default=REPOSITORY / "outputs" / "p1-a05-development-report",
    )
    parser.add_argument(
        "--training-dir",
        type=Path,
        default=REPOSITORY / "outputs" / "p1-a05-size-robustness",
    )
    parser.add_argument(
        "--g3-authorization",
        type=Path,
        default=REPOSITORY / "outputs" / "p1-06" / "g3_authorization.json",
    )
    parser.add_argument(
        "--independent-review",
        type=Path,
        default=REPOSITORY
        / "outputs"
        / "p1-a05-independent-review"
        / "p1_a05_independent_review.json",
    )


def _input_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "repository": args.repository,
        "contract_path": args.contract,
        "evaluation_contract_path": args.evaluation_contract,
        "preregister_path": args.preregister,
        "implementation_review_path": args.implementation_review,
        "comparison_path": args.comparison,
        "baseline_evidence_path": args.baseline_evidence,
        "candidate_evidence_path": args.candidate_evidence,
        "candidate_report_dir": args.candidate_report_dir,
        "training_dir": args.training_dir,
        "independent_review_path": args.independent_review,
        "g3_authorization_path": args.g3_authorization,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare, build, or verify the gated P1-06 model/result bundle"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-review")
    _add_input_arguments(prepare)
    prepare.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY / "outputs" / "p1-06" / "p1_06_review_request.json",
    )
    build = subparsers.add_parser("build")
    _add_input_arguments(build)
    build.add_argument(
        "--review-request",
        type=Path,
        default=REPOSITORY / "outputs" / "p1-06" / "p1_06_review_request.json",
    )
    build.add_argument(
        "--review-a",
        type=Path,
        default=REPOSITORY / "outputs" / "p1-06" / "p1_06_review_a.json",
    )
    build.add_argument(
        "--review-b",
        type=Path,
        default=REPOSITORY / "outputs" / "p1-06" / "p1_06_review_b.json",
    )
    build.add_argument("--output", type=Path, default=None)
    verify = subparsers.add_parser("verify")
    verify.add_argument("archive", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "verify":
            manifest = verify_model_result_bundle(args.archive)
            print(
                json.dumps(
                    {
                        "ok": True,
                        "candidate_commit": manifest["candidate_commit"],
                        "release_commit": manifest["release_commit"],
                        "file_count": len(manifest["files"]),
                        "sha256": _file_sha256(args.archive),
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        kwargs = _input_kwargs(args)
        if args.command == "prepare-review":
            result = prepare_review_request(output_path=args.output, **kwargs)
        else:
            output = args.output
            if output is None:
                commit = _clean_head(args.repository.resolve())
                output = (
                    args.repository
                    / "outputs"
                    / "release"
                    / f"trisched-model-results-{commit[:12]}.zip"
                )
            result = build_model_result_bundle(
                output_path=output,
                review_request_path=args.review_request,
                review_receipt_paths=[args.review_a, args.review_b],
                **kwargs,
            )
        print(result.resolve())
        if result.suffix == ".zip":
            print(f"sha256={_file_sha256(result)}")
        return 0
    except P106BundleError as error:
        print(
            json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
