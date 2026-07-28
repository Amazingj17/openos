from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from trisched import cli
from trisched.cli import load_config, run_pipeline
from trisched.env import ScheduleResult, run_policy
from trisched.evaluation import evaluate_schedulers
from trisched.policies import HeftPolicy
from trisched.scenario import Edge, Resource, Scenario, Task
from trisched.schedulers import (
    PolicySchedulerRunner,
    SchedulerAdapterError,
)


def scenario(identifier: str) -> Scenario:
    return Scenario(
        id=identifier,
        seed=91,
        tasks=(Task(0, 4.0), Task(1, 2.0)),
        resources=(
            Resource(0, "edge-0", "edge", 1.0),
            Resource(1, "cloud-0", "cloud", 2.0),
        ),
        edges=(Edge(0, 1, 1.0),),
        bandwidth=((1e9, 4.0), (4.0, 1e9)),
        latency=((0.0, 0.25), (0.25, 0.0)),
    )


class FlakyHeftRunner:
    name = "flaky"

    def __init__(self, failing_ids: set[str]) -> None:
        self.failing_ids = failing_ids

    def schedule(self, item: Scenario) -> ScheduleResult:
        if item.id in self.failing_ids:
            raise SchedulerAdapterError(
                "scheduler_timeout",
                "independent timeout injection",
                scheduler=self.name,
                scenario_id=item.id,
                details={"timeout_seconds": 0.1},
            )
        result = run_policy(item, HeftPolicy())
        return ScheduleResult(
            scenario_id=result.scenario_id,
            policy_name=self.name,
            entries=result.entries,
            makespan=result.makespan,
        )


class BrokenBaseline:
    name = "heft"

    def schedule(self, item: Scenario) -> ScheduleResult:
        raise RuntimeError(f"baseline failed for {item.id}")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tiny_config(output: Path) -> dict[str, object]:
    return {
        "seed": 71,
        "output_dir": str(output),
        "dataset": {
            "train_count": 2,
            "validation_count": 1,
            "test_count": 1,
            "task_range": [3, 4],
            "resource_count": 2,
            "edge_probability": 0.2,
        },
        "training": {
            "hidden_dim": 4,
            "imitation_epochs": 1,
            "reinforce_epochs": 1,
            "imitation_learning_rate": 0.005,
            "reinforce_learning_rate": 0.001,
            "reinforce_temperature": 2.0,
            "gradient_clip": 5.0,
        },
        "evaluation": {"random_seed": 5},
    }


def test_failed_instances_remain_in_rows_metrics_and_jsonl(
    tmp_path: Path,
) -> None:
    schedulers = [
        PolicySchedulerRunner("heft", HeftPolicy),
        FlakyHeftRunner({"case-fail"}),
    ]
    metrics, rows = evaluate_schedulers(
        [scenario("case-ok"), scenario("case-fail")],
        schedulers,
        "test",
        tmp_path,
        failure_penalty_ratio=7.0,
    )

    flaky = metrics["flaky"]
    assert flaky["count"] == 2
    assert flaky["success_count"] == 1
    assert flaky["failure_count"] == 1
    assert flaky["failure_rate"] == pytest.approx(0.5)
    assert flaky["valid_schedule_rate"] == pytest.approx(0.5)
    assert flaky["success_mean_ratio"] == pytest.approx(1.0)
    assert flaky["mean_ratio"] == pytest.approx(4.0)
    assert flaky["tie_rate_vs_heft"] == pytest.approx(0.5)
    assert flaky["loss_rate_vs_heft"] == pytest.approx(0.5)
    assert flaky["error_counts"] == {"scheduler_timeout": 1}

    failed = rows[1]
    assert failed["flaky_status"] == "failure"
    assert failed["flaky_makespan"] == ""
    assert failed["flaky_ratio"] == ""
    assert failed["flaky_score_ratio"] == pytest.approx(7.0)
    assert failed["flaky_penalty_applied"] is True
    assert failed["flaky_error_code"] == "scheduler_timeout"

    csv_rows = list(
        csv.DictReader(
            (tmp_path / "test_per_instance.csv").open(
                encoding="utf-8", newline=""
            )
        )
    )
    assert len(csv_rows) == 2
    assert csv_rows[1]["flaky_status"] == "failure"
    assert csv_rows[1]["flaky_score_ratio"] == "7.0"

    failure_lines = (
        tmp_path / "test_failures.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(failure_lines) == 1
    failure = json.loads(failure_lines[0])
    assert failure["scenario_id"] == "case-fail"
    assert failure["scheduler"] == "flaky"
    assert failure["error"]["code"] == "scheduler_timeout"
    assert failure["failure_penalty_ratio"] == 7.0

    examples = json.loads(
        (tmp_path / "test_example_schedule.json").read_text(encoding="utf-8")
    )
    assert examples["flaky"]["scenario_id"] == "case-ok"


def test_all_failures_have_null_success_metrics(tmp_path: Path) -> None:
    schedulers = [
        PolicySchedulerRunner("heft", HeftPolicy),
        FlakyHeftRunner({"case-1", "case-2"}),
    ]
    metrics, _ = evaluate_schedulers(
        [scenario("case-1"), scenario("case-2")],
        schedulers,
        "validation",
        tmp_path,
        failure_penalty_ratio=9.0,
    )
    flaky = metrics["flaky"]
    assert flaky["failure_rate"] == 1.0
    assert flaky["valid_schedule_rate"] == 0.0
    assert flaky["success_mean_makespan"] is None
    assert flaky["success_mean_ratio"] is None
    assert flaky["mean_ratio"] == 9.0


def test_heft_baseline_failure_is_not_scored(tmp_path: Path) -> None:
    with pytest.raises(SchedulerAdapterError) as captured:
        evaluate_schedulers(
            [scenario("baseline-case")],
            [BrokenBaseline(), FlakyHeftRunner(set())],
            "test",
            tmp_path,
        )
    assert captured.value.code == "scheduler_baseline_failed"
    assert captured.value.scheduler == "heft"
    assert captured.value.scenario_id == "baseline-case"
    assert captured.value.details["cause"]["code"] == "scheduler_execution_failed"
    assert not (tmp_path / "test_per_instance.csv").exists()


@pytest.mark.parametrize("value", [True, "10", 1.0, 0.0, float("inf")])
def test_failure_penalty_configuration_is_strict(
    tmp_path: Path, value: object
) -> None:
    config = tiny_config(tmp_path / "output")
    config["evaluation"]["failure_penalty_ratio"] = value  # type: ignore[index]
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="failure_penalty_ratio"):
        load_config(config_path)


def test_pipeline_manifest_hashes_every_declared_artifact(tmp_path: Path) -> None:
    output = tmp_path / "run"
    config = tiny_config(output)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    summary_path = run_pipeline(config_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert summary["format_version"] == 2
    assert summary["run_manifest"] == "run_manifest.json"
    assert summary["scoring"]["failure_penalty_ratio"] == 10.0
    assert manifest["mode"] == "pipeline"
    assert manifest["scoring"]["failure_penalty_ratio"] == 10.0
    assert len(manifest["code"]["commit"]) == 40
    assert manifest["inputs"]["dependency_lock"] is not None
    assert len(manifest["inputs"]["dependency_lock"]["sha256"]) == 64
    assert manifest["inputs"]["config"]["source_sha256"] == file_sha256(
        config_path
    )
    assert manifest["inputs"]["checkpoint"]["sha256"] == file_sha256(
        output / "masked_mlp.npz"
    )
    assert "run_manifest.json" not in manifest["artifacts"]
    assert (output / "validation_failures.jsonl").read_text(
        encoding="utf-8"
    ) == ""
    assert (output / "test_failures.jsonl").read_text(encoding="utf-8") == ""
    for name, metadata in manifest["artifacts"].items():
        artifact = output / name
        assert artifact.stat().st_size == metadata["bytes"]
        assert file_sha256(artifact) == metadata["sha256"]


def test_environment_git_provenance_includes_explicit_clean_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRISCHED_GIT_HEAD", "a" * 40)
    monkeypatch.setenv("TRISCHED_GIT_DIRTY", "false")
    assert cli._git_metadata() == {
        "commit": "a" * 40,
        "working_tree_dirty": False,
        "source": "TRISCHED_GIT_HEAD",
    }


def test_cli_reports_aggregated_failures_after_writing_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary_path = tmp_path / "evaluation_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "mode": "checkpoint_evaluation",
                "split": "test",
                "metrics": {
                    "heft": {"failure_count": 0},
                    "flaky": {
                        "failure_count": 2,
                        "failure_rate": 0.5,
                        "error_counts": {"scheduler_timeout": 2},
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(cli, "evaluate_checkpoint", lambda *args: summary_path)
    exit_code = cli.main(["evaluate"])
    captured = capsys.readouterr()
    assert exit_code == 3
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["error"]["code"] == "evaluation_contains_failures"
    assert payload["error"]["details"]["failure_count"] == 2
    assert payload["error"]["details"]["results"][0]["scheduler"] == "flaky"
