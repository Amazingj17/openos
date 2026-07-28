from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from trisched import cli
from trisched.env import run_policy, validate_schedule
from trisched.evaluation import evaluate_split
from trisched.policies import HeftPolicy
from trisched.scenario import Edge, Resource, Scenario, Task
from trisched.schedulers import (
    DEFAULT_SCHEDULER_NAMES,
    ExternalProcessScheduler,
    SchedulerAdapterError,
    SchedulerContext,
    build_scheduler_runners,
    create_default_scheduler_registry,
)


REPOSITORY = Path(__file__).resolve().parents[1]


class LearnedHeftPolicy(HeftPolicy):
    name = "masked_mlp"


def toy_scenario() -> Scenario:
    return Scenario(
        id="external-toy",
        seed=17,
        tasks=(Task(0, 4.0), Task(1, 3.0), Task(2, 5.0)),
        resources=(
            Resource(0, "edge-0", "edge", 1.0),
            Resource(1, "cloud-0", "cloud", 2.0),
        ),
        edges=(Edge(0, 2, 2.0), Edge(1, 2, 1.0)),
        bandwidth=((1e9, 4.0), (4.0, 1e9)),
        latency=((0.0, 0.25), (0.25, 0.0)),
    )


def write_script(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "scheduler.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


def external_runner(
    tmp_path: Path, source: str, **options: object
) -> ExternalProcessScheduler:
    script = write_script(tmp_path, source)
    return ExternalProcessScheduler(
        name="external_test",
        command=(sys.executable, str(script)),
        working_directory=tmp_path,
        **options,
    )


def test_reference_external_heft_matches_in_process_heft() -> None:
    scenario = toy_scenario()
    runner = ExternalProcessScheduler(
        name="external_heft",
        command=(
            sys.executable,
            "-m",
            "examples.external_heft_scheduler",
        ),
        timeout_seconds=10.0,
        working_directory=REPOSITORY,
    )
    external = runner.schedule(scenario)
    expected = run_policy(scenario, HeftPolicy())
    validate_schedule(scenario, external)
    assert external.policy_name == "external_heft"
    assert external.makespan == pytest.approx(expected.makespan)
    assert external.entries == expected.entries


def test_registry_and_unified_evaluator_include_external_scheduler(
    tmp_path: Path,
) -> None:
    registry = create_default_scheduler_registry()
    context = SchedulerContext(LearnedHeftPolicy(), 29, REPOSITORY)
    assert registry.names == tuple(sorted(DEFAULT_SCHEDULER_NAMES))
    assert registry.create("heft", context).name == "heft"
    with pytest.raises(SchedulerAdapterError) as duplicate:
        registry.register("heft", lambda item: registry.create("heft", item))
    assert duplicate.value.code == "scheduler_duplicate"


    with pytest.raises(SchedulerAdapterError) as unknown:
        registry.create("missing", context)
    assert unknown.value.code == "scheduler_unknown"

    specs: list[object] = list(DEFAULT_SCHEDULER_NAMES)
    specs.append(
        {
            "type": "external",
            "name": "external_heft",
            "command": [
                "{python}",
                "-m",
                "examples.external_heft_scheduler",
            ],
            "timeout_seconds": 10,
            "working_directory": str(REPOSITORY),
        }
    )
    output = tmp_path / "evaluation"
    metrics, rows = evaluate_split(
        [toy_scenario()],
        LearnedHeftPolicy(),
        "test",
        output,
        random_seed=29,
        scheduler_specs=specs,
        config_dir=tmp_path,
    )
    assert metrics["external_heft"]["count"] == 1
    assert metrics["external_heft"]["mean_ratio"] == pytest.approx(1.0)
    assert rows[0]["external_heft_ratio"] == pytest.approx(1.0)
    assert "external_heft_runtime_ms" in rows[0]
    examples = json.loads(
        (output / "test_example_schedule.json").read_text(encoding="utf-8")
    )
    assert examples["external_heft"]["policy_name"] == "external_heft"


def test_external_timeout_has_stable_code(tmp_path: Path) -> None:
    runner = external_runner(
        tmp_path,
        "import time\ntime.sleep(2)\n",
        timeout_seconds=0.05,
    )
    with pytest.raises(SchedulerAdapterError) as captured:
        runner.schedule(toy_scenario())
    assert captured.value.code == "scheduler_timeout"
    assert captured.value.scheduler == "external_test"
    assert captured.value.scenario_id == "external-toy"


def test_external_stdout_and_process_failures_have_stable_codes(
    tmp_path: Path,
) -> None:
    cases = [
        (
            "import sys\nsys.stderr.write('boom')\nraise SystemExit(7)\n",
            {},
            "scheduler_process_failed",
        ),
        ("print('not-json')\n", {}, "scheduler_invalid_json"),
        (
            "import sys\nsys.stdout.buffer.write(bytes([255]))\n",
            {},
            "scheduler_invalid_utf8",
        ),
        (
            "print('x' * 100)\n",
            {"max_output_bytes": 10},
            "scheduler_output_too_large",
        ),
    ]
    for index, (source, options, expected_code) in enumerate(cases):
        case_dir = tmp_path / str(index)
        case_dir.mkdir()
        runner = external_runner(case_dir, source, **options)
        with pytest.raises(SchedulerAdapterError) as captured:
            runner.schedule(toy_scenario())
        assert captured.value.code == expected_code
        assert captured.value.to_dict()["code"] == expected_code


def test_external_invalid_response_and_schedule_are_separate(
    tmp_path: Path,
) -> None:
    wrong_protocol_dir = tmp_path / "wrong-protocol"
    wrong_protocol_dir.mkdir()
    wrong_protocol = external_runner(
        wrong_protocol_dir,
        """
        import json
        import sys
        request = json.load(sys.stdin)
        print(json.dumps({
            "protocol_version": 99,
            "scheduler_name": request["scheduler_name"],
            "scenario_id": request["scenario"]["id"],
            "makespan": 1.0,
            "entries": [],
        }))
        """,
    )
    with pytest.raises(SchedulerAdapterError) as invalid_response:
        wrong_protocol.schedule(toy_scenario())
    assert invalid_response.value.code == "scheduler_invalid_response"
    assert invalid_response.value.details["path"] == "$.protocol_version"

    invalid_schedule_dir = tmp_path / "invalid-schedule"
    invalid_schedule_dir.mkdir()
    invalid_schedule = external_runner(
        invalid_schedule_dir,
        """
        import json
        import sys
        request = json.load(sys.stdin)
        print(json.dumps({
            "protocol_version": 1,
            "scheduler_name": request["scheduler_name"],
            "scenario_id": request["scenario"]["id"],
            "makespan": 1.0,
            "entries": [],
        }))
        """,
    )
    with pytest.raises(SchedulerAdapterError) as invalid_result:
        invalid_schedule.schedule(toy_scenario())
    assert invalid_result.value.code == "scheduler_invalid_schedule"
    assert "every task" in invalid_result.value.details["reason"]


def test_launch_failure_and_scheduler_spec_validation(tmp_path: Path) -> None:
    missing = ExternalProcessScheduler(
        name="external_test",
        command=(str(tmp_path / "does-not-exist"),),
        working_directory=tmp_path,
    )
    with pytest.raises(SchedulerAdapterError) as launch:
        missing.schedule(toy_scenario())
    assert launch.value.code == "scheduler_launch_failed"

    with pytest.raises(SchedulerAdapterError) as required:
        build_scheduler_runners(
            LearnedHeftPolicy(),
            random_seed=1,
            scheduler_specs=["heft"],
            config_dir=tmp_path,
        )
    assert required.value.code == "scheduler_invalid_spec"
    assert required.value.details["missing"] == ["masked_mlp"]

    duplicate_specs: list[object] = list(DEFAULT_SCHEDULER_NAMES)
    duplicate_specs.append({"type": "builtin", "name": "heft"})
    with pytest.raises(SchedulerAdapterError) as duplicate:
        build_scheduler_runners(
            LearnedHeftPolicy(),
            random_seed=1,
            scheduler_specs=duplicate_specs,
            config_dir=tmp_path,
        )
    assert duplicate.value.code == "scheduler_duplicate"


def test_pipeline_config_wires_external_scheduler(tmp_path: Path) -> None:
    output = tmp_path / "pipeline"
    schedulers: list[object] = list(DEFAULT_SCHEDULER_NAMES)
    schedulers.append(
        {
            "type": "external",
            "name": "external_heft",
            "command": [
                "{python}",
                "-m",
                "examples.external_heft_scheduler",
            ],
            "timeout_seconds": 10,
            "working_directory": str(REPOSITORY),
        }
    )
    config = {
        "seed": 31,
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
        "evaluation": {"random_seed": 5, "schedulers": schedulers},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    summary_path = cli.run_pipeline(config_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["validation"]["external_heft"]["count"] == 1
    assert summary["test"]["external_heft"]["mean_ratio"] == pytest.approx(1.0)


def test_cli_emits_scheduler_failure_as_one_json_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_pipeline(config: str, output: str | None) -> None:
        raise SchedulerAdapterError(
            "scheduler_timeout",
            "deadline exceeded",
            scheduler="external_test",
            scenario_id="case-1",
            details={"timeout_seconds": 1.0},
        )

    monkeypatch.setattr(cli, "run_pipeline", fail_pipeline)
    exit_code = cli.main(["pipeline", "--config", "unused.json"])
    captured = capsys.readouterr()
    assert exit_code == 3
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload == {
        "ok": False,
        "error": {
            "code": "scheduler_timeout",
            "message": "deadline exceeded",
            "scheduler": "external_test",
            "scenario_id": "case-1",
            "details": {"timeout_seconds": 1.0},
        },
    }
