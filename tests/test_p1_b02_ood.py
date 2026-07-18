from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from trisched import cli
from trisched.env import ScheduleResult, run_policy
from trisched.ood import (
    MATERIALIZATION_MANIFEST_NAME,
    OODWorkflowError,
    load_materialized_development_slices,
    materialize_development_slices,
    materialize_p1_b02_ood,
    produce_development_evidence,
    transform_ood_ccr,
    transform_ood_system,
)
from trisched.policies import HeftPolicy
from trisched.reporting import build_evaluation_report, canonical_json_sha256
from trisched.reporting import load_evaluation_contract
from trisched.scenario import Scenario, generate_scenario
from trisched.schedulers import PolicySchedulerRunner, SchedulerAdapterError


ROOT = Path(__file__).resolve().parents[1]
TRACKED_CONTRACT = ROOT / "configs" / "p1_b02_evaluation_contract.json"
TRACKED_BENCHMARK = ROOT / "data" / "benchmarks" / "stg-rnc50-hetero-v1.json"
TRACKED_DEVELOPMENT_MANIFEST = (
    ROOT / "data" / "benchmarks" / "p1-b02-development-slices-v1.json"
)


def write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def tiny_contract() -> dict[str, Any]:
    return {
        "format_version": 1,
        "contract_id": "p1_b02_ood_test_v1",
        "primary_policy": "masked_mlp",
        "reference_policy": "heft",
        "tie_tolerance": 1e-9,
        "failure_penalty_ratio": 10.0,
        "metrics": {
            "score_ratio": "policy_makespan_over_heft_makespan",
            "runtime_ms": "wall_clock_scheduler_call_only",
            "runtime_quantiles": [0.5, 0.95],
            "timeout_error_code": "scheduler_timeout",
            "failures_remain_in_denominator": True,
        },
        "bootstrap": {
            "samples": 64,
            "seed": 1707,
            "confidence_level": 0.95,
            "hierarchy": ["seed", "scenario"],
        },
        "policies": [
            {"id": "heft", "role": "baseline", "required_seeds": [0]},
            {
                "id": "random",
                "role": "baseline",
                "required_seeds": [11, 12],
            },
            {
                "id": "masked_mlp",
                "role": "primary",
                "required_seeds": [11, 12],
            },
        ],
        "slices": [
            {
                "id": "id_validation",
                "role": "development",
                "scenario_count": 2,
                "definition": "tiny frozen validation slice",
                "source": {
                    "benchmark_id": "tiny-benchmark-v1",
                    "split": "validation",
                    "purpose": "model_selection",
                },
            },
            {
                "id": "ood_size",
                "role": "development_ood",
                "scenario_count": 2,
                "definition": "tiny deterministic size slice",
                "source": {
                    "generator": "trisched.scenario.generate_scenario",
                    "scenario_id_template": "ood-size-{index:04d}",
                    "generator_seeds": [1001, 1002],
                    "task_count": 8,
                    "resource_count": 3,
                    "edge_probability": 0.2,
                },
            },
            {
                "id": "ood_ccr",
                "role": "development_ood",
                "scenario_count": 2,
                "definition": "tiny deterministic CCR slice",
                "source": {
                    "base_slice": "id_validation",
                    "scenario_id_template": "ood-ccr-{index:04d}",
                    "edge_data_scale": 2.0,
                    "off_diagonal_bandwidth_scale": 0.5,
                },
            },
            {
                "id": "ood_system",
                "role": "development_ood",
                "scenario_count": 2,
                "definition": "tiny deterministic system slice",
                "source": {
                    "base_slice": "id_validation",
                    "scenario_id_template": "ood-system-{index:04d}",
                    "resource_speed_multipliers": [0.65, 1.0, 1.35],
                    "off_diagonal_bandwidth_scale": 0.75,
                    "off_diagonal_latency_scale": 1.5,
                },
            },
            {
                "id": "public_test",
                "role": "final_test",
                "scenario_count": 2,
                "definition": "synthetic gate-only final slice",
                "source": {"benchmark_id": "tiny-benchmark-v1", "split": "test"},
            },
        ],
        "modes": {
            "development": [
                "id_validation",
                "ood_size",
                "ood_ccr",
                "ood_system",
            ],
            "final_test": [
                "id_validation",
                "ood_size",
                "ood_ccr",
                "ood_system",
                "public_test",
            ],
        },
        "test_gate": {
            "slice_id": "public_test",
            "policy": "one_time_final_only",
            "required_signers": ["A", "B"],
            "require_clean_commit": True,
            "receipt_name": "public_test_gate_receipt.json",
            "authorization_time_format": "utc_seconds_z",
        },
    }


def validation_scenarios() -> list[Scenario]:
    return [
        generate_scenario(
            seed=41 + index,
            task_count=6,
            resource_count=3,
            edge_probability=0.3,
            scenario_id=f"validation-{index:04d}",
        )
        for index in range(2)
    ]


def source_binding(scenarios: list[Scenario]) -> dict[str, Any]:
    entries = [
        {
            "source": f"rnc50_hetero/tiny-{index:04d}.json",
            "source_sha256": hashlib.sha256(f"source-{index}".encode()).hexdigest(),
            "scenario_id": scenario.id,
            "scenario_hash": scenario.content_hash(),
            "split_index": index,
        }
        for index, scenario in enumerate(scenarios)
    ]
    return {
        "benchmark_id": "tiny-benchmark-v1",
        "manifest_name": "tiny-manifest.json",
        "manifest_sha256": hashlib.sha256(b"tiny-manifest").hexdigest(),
        "split": "validation",
        "purpose": "model_selection",
        "scenario_hashes_sha256": canonical_json_sha256(
            [scenario.content_hash() for scenario in scenarios]
        ),
        "entries": entries,
    }


def materialized_fixture(tmp_path: Path) -> tuple[Path, Path, list[Scenario]]:
    contract_path = write_json(tmp_path / "contract.json", tiny_contract())
    scenarios = validation_scenarios()
    root = tmp_path / "materialized"
    manifest = materialize_development_slices(
        contract_path,
        scenarios,
        source_binding(scenarios),
        root,
    )
    return contract_path, manifest, scenarios


def test_tracked_development_manifest_is_frozen_and_source_bound() -> None:
    contract = load_evaluation_contract(TRACKED_CONTRACT)
    benchmark = json.loads(TRACKED_BENCHMARK.read_text(encoding="utf-8"))
    manifest = json.loads(TRACKED_DEVELOPMENT_MANIFEST.read_text(encoding="utf-8"))
    assert file_sha256(TRACKED_DEVELOPMENT_MANIFEST) == (
        "d9c6e22ee7991b5659a46b6503a98580e9f3262edc5381bc9a32b8cf44acf2be"
    )
    assert manifest["contract"] == {
        "contract_id": contract["contract_id"],
        "canonical_sha256": canonical_json_sha256(contract),
    }
    assert manifest["benchmark"] == {
        "benchmark_id": "stg-rnc50-hetero-trisched-v1",
        "manifest_name": "stg-rnc50-hetero-v1.json",
        "manifest_sha256": file_sha256(TRACKED_BENCHMARK),
        "split": "validation",
        "purpose": "model_selection",
        "scenario_hashes_sha256": (
            "64c5e67f00b7bab5c6e9f8c21561448452bd82525c83abf5972cb3ca59cf5386"
        ),
    }
    assert manifest["test_accessed"] is False
    assert manifest["public_test_materialized"] is False
    expected_sets = {
        "id_validation": (
            "fea76e27e4c0a90e1a3067b9367f7e22372c5463e23b65abaf70aa5695010b1b"
        ),
        "ood_size": (
            "6f525035c5bc07b0a9b6dfd9fc456ac3d2343d234d409c5259ae944967f0da1a"
        ),
        "ood_ccr": ("fef5aa2ce9526a9a91e1479c52f5bfbe30e34d9ce45e11311d231e2f1c6f53c8"),
        "ood_system": (
            "618942a58ccd2f20a1b97966f492fe00030225a7a02ba28c7f2a8fc54bc26431"
        ),
    }
    assert [item["slice_id"] for item in manifest["slices"]] == list(expected_sets)
    expected_validation_sources = [
        {
            "source": entry["source"],
            "source_sha256": entry["source_sha256"],
            "scenario_id": entry["scenario_id"],
            "scenario_hash": entry["scenario_hash"],
            "split_index": entry["split_index"],
        }
        for entry in benchmark["entries"]
        if entry["split"] == "validation"
    ]
    actual_validation_sources = [
        {key: value for key, value in entry["provenance"].items() if key != "kind"}
        for entry in manifest["slices"][0]["entries"]
    ]
    assert actual_validation_sources == expected_validation_sources
    observed_ids: set[str] = set()
    observed_hashes: set[str] = set()
    for item in manifest["slices"]:
        assert item["scenario_count"] == 30
        assert len(item["entries"]) == 30
        pairs = []
        for index, entry in enumerate(item["entries"]):
            assert entry["index"] == index
            assert entry["file"] == f"slices/{item['slice_id']}/{index:04d}.json"
            assert entry["resource_count"] == 3
            assert entry["scenario_id"] not in observed_ids
            assert entry["scenario_hash"] not in observed_hashes
            observed_ids.add(entry["scenario_id"])
            observed_hashes.add(entry["scenario_hash"])
            pairs.append(
                {
                    "scenario_id": entry["scenario_id"],
                    "scenario_hash": entry["scenario_hash"],
                }
            )
        assert item["scenario_set_sha256"] == expected_sets[item["slice_id"]]
        assert item["scenario_set_sha256"] == canonical_json_sha256(pairs)
    assert len(observed_ids) == 120
    assert len(observed_hashes) == 120


def test_materializer_is_deterministic_and_applies_exact_transforms(
    tmp_path: Path,
) -> None:
    contract_path = write_json(tmp_path / "contract.json", tiny_contract())
    scenarios = validation_scenarios()
    binding = source_binding(scenarios)
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_manifest_path = materialize_development_slices(
        contract_path, scenarios, binding, first_root
    )
    second_manifest_path = materialize_development_slices(
        contract_path, scenarios, binding, second_root
    )
    assert tree_bytes(first_root) == tree_bytes(second_root)
    assert first_manifest_path.name == MATERIALIZATION_MANIFEST_NAME
    assert first_manifest_path.read_bytes() == second_manifest_path.read_bytes()

    loaded, manifest = load_materialized_development_slices(contract_path, first_root)
    assert list(loaded) == ["id_validation", "ood_size", "ood_ccr", "ood_system"]
    assert manifest["test_accessed"] is False
    assert manifest["public_test_materialized"] is False
    assert all(item["slice_id"] != "public_test" for item in manifest["slices"])
    assert [scenario.id for scenario in loaded["ood_size"]] == [
        "ood-size-0000",
        "ood-size-0001",
    ]
    assert [scenario.seed for scenario in loaded["ood_size"]] == [1001, 1002]
    assert all(scenario.task_count == 8 for scenario in loaded["ood_size"])

    base = loaded["id_validation"][0]
    ccr = loaded["ood_ccr"][0]
    assert ccr.tasks == base.tasks
    assert ccr.resources == base.resources
    assert ccr.latency == base.latency
    assert [(edge.source, edge.target) for edge in ccr.edges] == [
        (edge.source, edge.target) for edge in base.edges
    ]
    assert [edge.data for edge in ccr.edges] == pytest.approx(
        [edge.data * 2.0 for edge in base.edges]
    )
    for row in range(base.resource_count):
        for column in range(base.resource_count):
            expected = (
                base.bandwidth[row][column]
                if row == column
                else base.bandwidth[row][column] * 0.5
            )
            assert ccr.bandwidth[row][column] == pytest.approx(expected)

    system = loaded["ood_system"][0]
    assert system.tasks == base.tasks
    assert system.edges == base.edges
    assert [resource.speed for resource in system.resources] == pytest.approx(
        [
            base.resources[index].speed * multiplier
            for index, multiplier in enumerate([0.65, 1.0, 1.35])
        ]
    )
    for row in range(base.resource_count):
        for column in range(base.resource_count):
            expected_bandwidth = (
                base.bandwidth[row][column]
                if row == column
                else base.bandwidth[row][column] * 0.75
            )
            expected_latency = (
                base.latency[row][column]
                if row == column
                else base.latency[row][column] * 1.5
            )
            assert system.bandwidth[row][column] == pytest.approx(expected_bandwidth)
            assert system.latency[row][column] == pytest.approx(expected_latency)


def test_transform_functions_do_not_mutate_base_scenario() -> None:
    base = validation_scenarios()[0]
    before = base.to_dict()
    transform_ood_ccr(
        base,
        scenario_id="ood-ccr-0000",
        edge_data_scale=2.0,
        off_diagonal_bandwidth_scale=0.5,
    )
    transform_ood_system(
        base,
        scenario_id="ood-system-0000",
        resource_speed_multipliers=[0.65, 1.0, 1.35],
        off_diagonal_bandwidth_scale=0.75,
        off_diagonal_latency_scale=1.5,
    )
    assert base.to_dict() == before


def test_materializer_rejects_source_drift_and_existing_destination(
    tmp_path: Path,
) -> None:
    contract_path = write_json(tmp_path / "contract.json", tiny_contract())
    scenarios = validation_scenarios()
    binding = source_binding(scenarios)
    binding["entries"][0]["scenario_hash"] = "f" * 64
    with pytest.raises(OODWorkflowError) as captured:
        materialize_development_slices(
            contract_path, scenarios, binding, tmp_path / "drift"
        )
    assert captured.value.code == "ood_source"
    assert captured.value.path.endswith("scenario_hash")
    assert not (tmp_path / "drift").exists()

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    with pytest.raises(OODWorkflowError) as occupied_error:
        materialize_development_slices(
            contract_path,
            scenarios,
            source_binding(scenarios),
            occupied,
        )
    assert occupied_error.value.code == "ood_output_exists"


def test_loader_rejects_scenario_file_and_manifest_test_tampering(
    tmp_path: Path,
) -> None:
    contract_path, manifest_path, _ = materialized_fixture(tmp_path)
    root = manifest_path.parent
    scenario_path = root / "slices" / "ood_size" / "0000.json"
    scenario_path.write_bytes(scenario_path.read_bytes() + b" ")
    with pytest.raises(OODWorkflowError) as captured:
        load_materialized_development_slices(contract_path, root)
    assert captured.value.code == "ood_file_hash"

    other_root = tmp_path / "other"
    contract_path, other_manifest_path, _ = materialized_fixture(other_root)
    manifest = json.loads(other_manifest_path.read_text(encoding="utf-8"))
    manifest["test_accessed"] = True
    write_json(other_manifest_path, manifest)
    with pytest.raises(OODWorkflowError) as gate:
        load_materialized_development_slices(contract_path, other_manifest_path.parent)
    assert gate.value.code == "ood_test_gate"


def test_loader_rejects_benchmark_and_provenance_tampering(tmp_path: Path) -> None:
    contract_path, manifest_path, _ = materialized_fixture(tmp_path / "benchmark")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["benchmark"]["scenario_hashes_sha256"] = "f" * 64
    write_json(manifest_path, manifest)
    with pytest.raises(OODWorkflowError) as benchmark_error:
        load_materialized_development_slices(contract_path, manifest_path.parent)
    assert benchmark_error.value.code == "ood_source"
    assert benchmark_error.value.path == "$source_binding.scenario_hashes_sha256"

    contract_path, manifest_path, _ = materialized_fixture(tmp_path / "id")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["slices"][0]["entries"][0]["provenance"]["scenario_hash"] = "f" * 64
    write_json(manifest_path, manifest)
    with pytest.raises(OODWorkflowError) as source_error:
        load_materialized_development_slices(contract_path, manifest_path.parent)
    assert source_error.value.code == "ood_source"
    assert source_error.value.path.endswith("scenario_hash")

    contract_path, manifest_path, _ = materialized_fixture(tmp_path / "transform")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["slices"][2]["entries"][0]["provenance"]["base_scenario_id"] = "wrong-base"
    write_json(manifest_path, manifest)
    with pytest.raises(OODWorkflowError) as transform_error:
        load_materialized_development_slices(contract_path, manifest_path.parent)
    assert transform_error.value.code == "ood_source"
    assert transform_error.value.path.endswith("provenance")


def test_production_wrapper_loads_validation_with_model_selection_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract_path = write_json(tmp_path / "contract.json", tiny_contract())
    scenarios = validation_scenarios()
    entries = [
        {
            **entry,
            "split": "validation",
        }
        for entry in source_binding(scenarios)["entries"]
    ]
    benchmark = {
        "benchmark_id": "tiny-benchmark-v1",
        "splits": {
            "validation": {
                "scenario_hashes_sha256": canonical_json_sha256(
                    [scenario.content_hash() for scenario in scenarios]
                )
            }
        },
        "entries": entries,
    }
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr("trisched.ood.load_benchmark_manifest", lambda path: benchmark)

    def fake_load_split(
        root: str | Path,
        manifest: str | Path,
        split: str,
        *,
        purpose: str,
    ) -> list[Scenario]:
        calls.append((split, purpose))
        return scenarios

    monkeypatch.setattr("trisched.ood.load_frozen_split", fake_load_split)
    benchmark_path = write_json(tmp_path / "benchmark.json", benchmark)
    manifest_path = materialize_p1_b02_ood(
        contract_path,
        tmp_path / "raw",
        benchmark_path,
        tmp_path / "out",
    )
    assert manifest_path.exists()
    assert calls == [("validation", "model_selection")]


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class RenamedHeftRunner:
    def __init__(self, name: str, clock: FakeClock | None = None) -> None:
        self.name = name
        self.clock = clock

    def schedule(self, scenario: Scenario) -> ScheduleResult:
        if self.clock is not None:
            self.clock.value += 0.002
        result = run_policy(scenario, HeftPolicy())
        return ScheduleResult(
            scenario_id=result.scenario_id,
            policy_name=self.name,
            entries=result.entries,
            makespan=result.makespan,
        )


class FailingRunner:
    def __init__(self, name: str, code: str = "scheduler_timeout") -> None:
        self.name = name
        self.code = code

    def schedule(self, scenario: Scenario) -> ScheduleResult:
        raise SchedulerAdapterError(
            self.code,
            "injected scheduler failure",
            scheduler=self.name,
            scenario_id=scenario.id,
        )


class IllegalActionPolicy:
    def __init__(self, name: str) -> None:
        self.name = name

    def select_action(self, env: Any) -> tuple[int, int]:
        return (-1, -1)


class WrongTypeRunner:
    def __init__(self, name: str) -> None:
        self.name = name

    def schedule(self, scenario: Scenario) -> Any:
        return {"scenario_id": scenario.id, "not": "a ScheduleResult"}


def test_read_only_evidence_uses_scheduler_only_timer_and_builds_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract_path, manifest_path, _ = materialized_fixture(tmp_path)
    root = manifest_path.parent
    before = tree_bytes(root)
    clock = FakeClock()

    def validator(scenario: Scenario, result: ScheduleResult) -> None:
        clock.value += 1.0

    monkeypatch.setattr("trisched.ood.validate_schedule", validator)
    monkeypatch.setattr("trisched.ood.validate_schedule_independent", validator)

    def provider(policy: str, seed: int) -> RenamedHeftRunner:
        return RenamedHeftRunner(policy, clock)

    evidence_path = produce_development_evidence(
        contract_path,
        root,
        provider,
        tmp_path / "development-evidence.json",
        code={"commit": "a" * 40, "working_tree_dirty": False},
        clock=clock,
    )
    assert tree_bytes(root) == before
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["test_accessed"] is False
    assert evidence["producer"]["training_started"] is False
    assert evidence["producer"]["public_test_loaded"] is False
    assert evidence["producer"]["runtime_scope"] == (
        "wall_clock_runner_schedule_call_only"
    )
    assert len(evidence["records"]) == 40
    assert all(row["runtime_ms"] == pytest.approx(2.0) for row in evidence["records"])
    assert all(row["score_ratio"] == pytest.approx(1.0) for row in evidence["records"])
    report_path = build_evaluation_report(
        contract_path,
        evidence_path,
        tmp_path / "report",
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["report_scope"] == "development"
    assert report["gate"]["release_publishable"] is False


def test_evidence_retains_scheduler_failure_and_rejects_reference_failure(
    tmp_path: Path,
) -> None:
    contract_path, manifest_path, _ = materialized_fixture(tmp_path)
    root = manifest_path.parent

    def failing_random_provider(policy: str, seed: int) -> Any:
        if policy == "random":
            return FailingRunner(policy)
        return RenamedHeftRunner(policy)

    evidence_path = produce_development_evidence(
        contract_path,
        root,
        failing_random_provider,
        tmp_path / "failure-evidence.json",
        code={"commit": "b" * 40, "working_tree_dirty": False},
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    random_rows = [row for row in evidence["records"] if row["policy"] == "random"]
    assert random_rows
    assert all(row["status"] == "failure" for row in random_rows)
    assert all(row["score_ratio"] == 10.0 for row in random_rows)
    assert all(row["error_code"] == "scheduler_timeout" for row in random_rows)

    output = tmp_path / "reference-failure.json"

    def failing_heft_provider(policy: str, seed: int) -> Any:
        if policy == "heft":
            return FailingRunner(policy, "scheduler_execution_failed")
        return RenamedHeftRunner(policy)

    with pytest.raises(OODWorkflowError) as captured:
        produce_development_evidence(
            contract_path,
            root,
            failing_heft_provider,
            output,
            code={"commit": "c" * 40, "working_tree_dirty": False},
        )
    assert captured.value.code == "ood_reference_failed"
    assert not output.exists()


def test_evidence_counts_adapter_and_in_process_illegal_schedules(
    tmp_path: Path,
) -> None:
    contract_path, manifest_path, _ = materialized_fixture(tmp_path)
    root = manifest_path.parent

    def provider(policy: str, seed: int) -> Any:
        if policy == "random":
            return FailingRunner(policy, "scheduler_invalid_schedule")
        if policy == "masked_mlp":
            return PolicySchedulerRunner(
                policy,
                lambda: IllegalActionPolicy(policy),
            )
        return RenamedHeftRunner(policy)

    evidence_path = produce_development_evidence(
        contract_path,
        root,
        provider,
        tmp_path / "illegal-evidence.json",
        code={"commit": "e" * 40, "working_tree_dirty": False},
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    for policy in ("random", "masked_mlp"):
        rows = [row for row in evidence["records"] if row["policy"] == policy]
        assert len(rows) == 16
        assert all(row["status"] == "failure" for row in rows)
        assert all(row["error_code"] == "scheduler_invalid_schedule" for row in rows)
        assert all(row["illegal_action_count"] == 1 for row in rows)
        assert all(row["score_ratio"] == 10.0 for row in rows)

    report_path = build_evaluation_report(
        contract_path,
        evidence_path,
        tmp_path / "illegal-report",
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for slice_report in report["slices"]:
        for policy in ("random", "masked_mlp"):
            metrics = slice_report["policies"][policy]
            assert metrics["failure_count"] == 4
            assert metrics["illegal_action_count"] == 4
            assert metrics["illegal_record_count"] == 4


def test_evidence_structures_wrong_result_type_and_fails_reference(
    tmp_path: Path,
) -> None:
    contract_path, manifest_path, _ = materialized_fixture(tmp_path)
    root = manifest_path.parent

    def wrong_random_provider(policy: str, seed: int) -> Any:
        if policy == "random":
            return WrongTypeRunner(policy)
        return RenamedHeftRunner(policy)

    evidence_path = produce_development_evidence(
        contract_path,
        root,
        wrong_random_provider,
        tmp_path / "wrong-type-evidence.json",
        code={"commit": "f" * 40, "working_tree_dirty": False},
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    rows = [row for row in evidence["records"] if row["policy"] == "random"]
    assert len(rows) == 16
    assert all(row["status"] == "failure" for row in rows)
    assert all(row["error_code"] == "scheduler_invalid_response" for row in rows)
    assert all(row["illegal_action_count"] == 0 for row in rows)
    assert all(row["score_ratio"] == 10.0 for row in rows)

    output = tmp_path / "wrong-reference.json"

    def wrong_heft_provider(policy: str, seed: int) -> Any:
        if policy == "heft":
            return WrongTypeRunner(policy)
        return RenamedHeftRunner(policy)

    with pytest.raises(OODWorkflowError) as captured:
        produce_development_evidence(
            contract_path,
            root,
            wrong_heft_provider,
            output,
            code={"commit": "0" * 40, "working_tree_dirty": False},
        )
    assert captured.value.code == "ood_reference_failed"
    assert captured.value.details["error"]["code"] == "scheduler_invalid_response"
    assert not output.exists()


def test_evidence_rejects_mismatched_runner_and_existing_output(
    tmp_path: Path,
) -> None:
    contract_path, manifest_path, _ = materialized_fixture(tmp_path)
    root = manifest_path.parent

    with pytest.raises(OODWorkflowError) as mismatched:
        produce_development_evidence(
            contract_path,
            root,
            lambda policy, seed: RenamedHeftRunner("wrong_name"),
            tmp_path / "mismatch.json",
            code={"commit": "d" * 40, "working_tree_dirty": False},
        )
    assert mismatched.value.code == "ood_runner_provider"

    occupied = write_json(tmp_path / "occupied.json", {})
    with pytest.raises(OODWorkflowError) as existing:
        produce_development_evidence(
            contract_path,
            root,
            lambda policy, seed: RenamedHeftRunner(policy),
            occupied,
            code={"commit": "d" * 40, "working_tree_dirty": False},
        )
    assert existing.value.code == "ood_output_exists"


def test_materialize_ood_cli_and_structured_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = write_json(tmp_path / MATERIALIZATION_MANIFEST_NAME, {})
    calls: list[tuple[str, str, str, str]] = []

    def fake_materialize(
        contract: str,
        raw_root: str,
        benchmark_manifest: str,
        output: str,
    ) -> Path:
        calls.append((contract, raw_root, benchmark_manifest, output))
        return manifest

    monkeypatch.setattr(cli, "materialize_p1_b02_ood", fake_materialize)
    exit_code = cli.main(
        [
            "materialize-ood",
            "--contract",
            "contract.json",
            "--raw-root",
            "raw",
            "--benchmark-manifest",
            "benchmark.json",
            "--output",
            "out",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert Path(captured.out.strip()) == manifest.resolve()
    assert calls == [("contract.json", "raw", "benchmark.json", "out")]

    def fail(*args: Any) -> Path:
        raise OODWorkflowError("ood_source", "$benchmark", "injected")

    monkeypatch.setattr(cli, "materialize_p1_b02_ood", fail)
    exit_code = cli.main(["materialize-ood"])
    captured = capsys.readouterr()
    assert exit_code == 5
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"]["code"] == "ood_source"
    assert payload["error"]["path"] == "$benchmark"
