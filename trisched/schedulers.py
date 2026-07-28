from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .env import (
    IllegalActionError,
    ScheduleEntry,
    ScheduleResult,
    run_policy,
    validate_schedule,
)
from .oracle import validate_schedule_independent
from .policies import (
    CpopPolicy,
    GreedyEarliestFinishPolicy,
    HeftPolicy,
    RandomPolicy,
)
from .scenario import Scenario


EXTERNAL_SCHEDULER_PROTOCOL_VERSION = 1
DEFAULT_SCHEDULER_NAMES = (
    "heft",
    "cpop",
    "greedy_eft",
    "random",
    "masked_mlp",
)
_SCHEDULER_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_MAX_DIAGNOSTIC_CHARS = 2_000


class SchedulerAdapterError(RuntimeError):
    """Stable machine-readable failure raised by scheduler adapters."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        scheduler: str | None = None,
        scenario_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.scheduler = scheduler
        self.scenario_id = scenario_id
        self.details = dict(details or {})
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.scheduler is not None:
            payload["scheduler"] = self.scheduler
        if self.scenario_id is not None:
            payload["scenario_id"] = self.scenario_id
        if self.details:
            payload["details"] = self.details
        return payload


class SchedulerRunner(Protocol):
    name: str

    def schedule(self, scenario: Scenario) -> ScheduleResult:
        ...


@dataclass(frozen=True)
class PolicySchedulerRunner:
    """Adapt an in-process step policy to the whole-scenario runner contract."""

    name: str
    policy_factory: Callable[[], Any]

    def schedule(self, scenario: Scenario) -> ScheduleResult:
        policy = self.policy_factory()
        policy_name = getattr(policy, "name", None)
        if policy_name != self.name:
            raise SchedulerAdapterError(
                "scheduler_invalid_spec",
                "policy factory returned a scheduler with a different name",
                scheduler=self.name,
                scenario_id=scenario.id,
                details={"actual_name": policy_name},
            )
        try:
            return run_policy(scenario, policy)
        except IllegalActionError as error:
            raise SchedulerAdapterError(
                "scheduler_invalid_schedule",
                "in-process policy returned an illegal action",
                scheduler=self.name,
                scenario_id=scenario.id,
                details={
                    "task_id": error.task_id,
                    "resource_id": error.resource_id,
                    "reason": str(error),
                },
            ) from error


@dataclass(frozen=True)
class SchedulerContext:
    learned_policy: Any
    random_seed: int
    config_dir: Path


SchedulerFactory = Callable[[SchedulerContext], SchedulerRunner]


class SchedulerRegistry:
    """Registry for in-process schedulers used by the unified evaluator."""

    def __init__(self) -> None:
        self._factories: dict[str, SchedulerFactory] = {}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def register(self, name: str, factory: SchedulerFactory) -> None:
        _validate_scheduler_name(name)
        if name in self._factories:
            raise SchedulerAdapterError(
                "scheduler_duplicate",
                f"scheduler {name!r} is already registered",
                scheduler=name,
            )
        if not callable(factory):
            raise SchedulerAdapterError(
                "scheduler_invalid_spec",
                "scheduler factory must be callable",
                scheduler=name,
            )
        self._factories[name] = factory

    def create(self, name: str, context: SchedulerContext) -> SchedulerRunner:
        _validate_scheduler_name(name)
        factory = self._factories.get(name)
        if factory is None:
            raise SchedulerAdapterError(
                "scheduler_unknown",
                f"scheduler {name!r} is not registered",
                scheduler=name,
                details={"available": list(self.names)},
            )
        runner = factory(context)
        if runner.name != name:
            raise SchedulerAdapterError(
                "scheduler_invalid_spec",
                "scheduler factory returned a runner with a different name",
                scheduler=name,
                details={"actual_name": runner.name},
            )
        return runner


def _validate_scheduler_name(name: Any) -> str:
    if not isinstance(name, str) or _SCHEDULER_NAME.fullmatch(name) is None:
        raise SchedulerAdapterError(
            "scheduler_invalid_spec",
            "scheduler name must match ^[a-z][a-z0-9_-]{0,63}$",
            scheduler=name if isinstance(name, str) else None,
        )
    return name


def _learned_runner(context: SchedulerContext) -> SchedulerRunner:
    policy = context.learned_policy
    if policy is None or getattr(policy, "name", None) != "masked_mlp":
        raise SchedulerAdapterError(
            "scheduler_invalid_spec",
            "masked_mlp requires a loaded learned policy",
            scheduler="masked_mlp",
        )
    return PolicySchedulerRunner("masked_mlp", lambda: policy)


def create_default_scheduler_registry() -> SchedulerRegistry:
    registry = SchedulerRegistry()
    registry.register("heft", lambda context: PolicySchedulerRunner("heft", HeftPolicy))
    registry.register("cpop", lambda context: PolicySchedulerRunner("cpop", CpopPolicy))
    registry.register(
        "greedy_eft",
        lambda context: PolicySchedulerRunner("greedy_eft", GreedyEarliestFinishPolicy),
    )
    registry.register(
        "random",
        lambda context: PolicySchedulerRunner(
            "random", lambda: RandomPolicy(seed=context.random_seed)
        ),
    )
    registry.register("masked_mlp", _learned_runner)
    return registry


DEFAULT_SCHEDULER_REGISTRY = create_default_scheduler_registry()


def _diagnostic_excerpt(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = value
    return text[:_MAX_DIAGNOSTIC_CHARS]


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON constant {value} is not allowed")


def _response_error(
    scheduler: str, scenario_id: str, path: str, message: str
) -> SchedulerAdapterError:
    return SchedulerAdapterError(
        "scheduler_invalid_response",
        message,
        scheduler=scheduler,
        scenario_id=scenario_id,
        details={"path": path},
    )


def _finite_number(value: Any, scheduler: str, scenario_id: str, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _response_error(
            scheduler, scenario_id, path, "value must be a JSON number"
        )
    number = float(value)
    if not math.isfinite(number):
        raise _response_error(scheduler, scenario_id, path, "value must be finite")
    return number


def _integer(value: Any, scheduler: str, scenario_id: str, path: str) -> int:
    if type(value) is not int:
        raise _response_error(
            scheduler, scenario_id, path, "value must be a JSON integer"
        )
    return value


def _parse_external_response(
    payload: Any, scheduler: str, scenario: Scenario
) -> ScheduleResult:
    scenario_id = scenario.id
    if not isinstance(payload, dict):
        raise _response_error(
            scheduler, scenario_id, "$", "response must be a JSON object"
        )
    required = {
        "protocol_version",
        "scheduler_name",
        "scenario_id",
        "makespan",
        "entries",
    }
    keys = set(payload)
    if keys != required:
        raise _response_error(
            scheduler,
            scenario_id,
            "$",
            f"response keys must be exactly {sorted(required)}",
        )
    version = payload["protocol_version"]
    if type(version) is not int or version != EXTERNAL_SCHEDULER_PROTOCOL_VERSION:
        raise _response_error(
            scheduler,
            scenario_id,
            "$.protocol_version",
            f"protocol_version must be {EXTERNAL_SCHEDULER_PROTOCOL_VERSION}",
        )
    if payload["scheduler_name"] != scheduler:
        raise _response_error(
            scheduler,
            scenario_id,
            "$.scheduler_name",
            "scheduler_name does not match the configured name",
        )
    if payload["scenario_id"] != scenario_id:
        raise _response_error(
            scheduler,
            scenario_id,
            "$.scenario_id",
            "scenario_id does not match the request",
        )
    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list):
        raise _response_error(
            scheduler, scenario_id, "$.entries", "entries must be a JSON array"
        )
    entries: list[ScheduleEntry] = []
    entry_keys = {"task_id", "resource_id", "start", "finish"}
    for index, raw_entry in enumerate(raw_entries):
        path = f"$.entries[{index}]"
        if not isinstance(raw_entry, dict) or set(raw_entry) != entry_keys:
            raise _response_error(
                scheduler,
                scenario_id,
                path,
                f"entry keys must be exactly {sorted(entry_keys)}",
            )
        entries.append(
            ScheduleEntry(
                task_id=_integer(
                    raw_entry["task_id"], scheduler, scenario_id, f"{path}.task_id"
                ),
                resource_id=_integer(
                    raw_entry["resource_id"],
                    scheduler,
                    scenario_id,
                    f"{path}.resource_id",
                ),
                start=_finite_number(
                    raw_entry["start"], scheduler, scenario_id, f"{path}.start"
                ),
                finish=_finite_number(
                    raw_entry["finish"], scheduler, scenario_id, f"{path}.finish"
                ),
            )
        )
    makespan = _finite_number(payload["makespan"], scheduler, scenario_id, "$.makespan")
    return ScheduleResult(
        scenario_id=scenario_id,
        policy_name=scheduler,
        entries=tuple(entries),
        makespan=makespan,
    )


@dataclass(frozen=True)
class ExternalProcessScheduler:
    """Run one trusted external process per scenario using JSON over stdio."""

    name: str
    command: tuple[str, ...]
    timeout_seconds: float = 30.0
    working_directory: Path | None = None
    max_output_bytes: int = 4 * 1024 * 1024

    def __post_init__(self) -> None:
        _validate_scheduler_name(self.name)
        if isinstance(self.command, (str, bytes)) or not isinstance(
            self.command, Sequence
        ):
            raise SchedulerAdapterError(
                "scheduler_invalid_spec",
                "external command must be a non-empty string array",
                scheduler=self.name,
            )
        command = tuple(self.command)
        if not command or any(
            not isinstance(part, str) or not part for part in command
        ):
            raise SchedulerAdapterError(
                "scheduler_invalid_spec",
                "external command must be a non-empty string array",
                scheduler=self.name,
            )
        if isinstance(self.timeout_seconds, bool) or not isinstance(
            self.timeout_seconds, (int, float)
        ):
            raise SchedulerAdapterError(
                "scheduler_invalid_spec",
                "timeout_seconds must be a positive finite number",
                scheduler=self.name,
            )
        timeout = float(self.timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0:
            raise SchedulerAdapterError(
                "scheduler_invalid_spec",
                "timeout_seconds must be a positive finite number",
                scheduler=self.name,
            )
        if type(self.max_output_bytes) is not int or self.max_output_bytes <= 0:
            raise SchedulerAdapterError(
                "scheduler_invalid_spec",
                "max_output_bytes must be a positive integer",
                scheduler=self.name,
            )
        working_directory = Path(self.working_directory or Path.cwd()).resolve()
        if not working_directory.is_dir():
            raise SchedulerAdapterError(
                "scheduler_invalid_spec",
                "working_directory must exist and be a directory",
                scheduler=self.name,
                details={"working_directory": str(working_directory)},
            )
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "timeout_seconds", timeout)
        object.__setattr__(self, "working_directory", working_directory)

    def schedule(self, scenario: Scenario) -> ScheduleResult:
        request = json.dumps(
            {
                "protocol_version": EXTERNAL_SCHEDULER_PROTOCOL_VERSION,
                "scheduler_name": self.name,
                "scenario": scenario.to_dict(),
            },
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        try:
            completed = subprocess.run(
                self.command,
                input=request,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.working_directory,
                timeout=self.timeout_seconds,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as error:
            raise SchedulerAdapterError(
                "scheduler_timeout",
                f"external scheduler exceeded {self.timeout_seconds:g} seconds",
                scheduler=self.name,
                scenario_id=scenario.id,
                details={
                    "timeout_seconds": self.timeout_seconds,
                    "stderr": _diagnostic_excerpt(error.stderr),
                },
            ) from error
        except OSError as error:
            raise SchedulerAdapterError(
                "scheduler_launch_failed",
                "external scheduler process could not be started",
                scheduler=self.name,
                scenario_id=scenario.id,
                details={"reason": str(error)},
            ) from error

        if completed.returncode != 0:
            raise SchedulerAdapterError(
                "scheduler_process_failed",
                f"external scheduler exited with code {completed.returncode}",
                scheduler=self.name,
                scenario_id=scenario.id,
                details={
                    "exit_code": completed.returncode,
                    "stderr": _diagnostic_excerpt(completed.stderr),
                },
            )
        if len(completed.stdout) > self.max_output_bytes:
            raise SchedulerAdapterError(
                "scheduler_output_too_large",
                "external scheduler stdout exceeded max_output_bytes",
                scheduler=self.name,
                scenario_id=scenario.id,
                details={
                    "actual_bytes": len(completed.stdout),
                    "max_output_bytes": self.max_output_bytes,
                },
            )
        try:
            stdout = completed.stdout.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise SchedulerAdapterError(
                "scheduler_invalid_utf8",
                "external scheduler stdout is not UTF-8",
                scheduler=self.name,
                scenario_id=scenario.id,
            ) from error
        try:
            payload = json.loads(stdout, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError) as error:
            details: dict[str, Any] = {"stdout_excerpt": stdout[:_MAX_DIAGNOSTIC_CHARS]}
            if isinstance(error, json.JSONDecodeError):
                details.update({"line": error.lineno, "column": error.colno})
            raise SchedulerAdapterError(
                "scheduler_invalid_json",
                "external scheduler stdout is not strict JSON",
                scheduler=self.name,
                scenario_id=scenario.id,
                details=details,
            ) from error

        result = _parse_external_response(payload, self.name, scenario)
        try:
            validate_schedule(scenario, result)
            validate_schedule_independent(scenario, result)
        except ValueError as error:
            raise SchedulerAdapterError(
                "scheduler_invalid_schedule",
                "external scheduler returned an invalid schedule",
                scheduler=self.name,
                scenario_id=scenario.id,
                details={"reason": str(error)},
            ) from error
        return result


def _external_runner_from_spec(
    spec: Mapping[str, Any], context: SchedulerContext
) -> ExternalProcessScheduler:
    allowed = {
        "type",
        "name",
        "command",
        "timeout_seconds",
        "working_directory",
        "max_output_bytes",
    }
    unknown = set(spec) - allowed
    missing = {"name", "command"} - set(spec)
    if unknown or missing:
        raise SchedulerAdapterError(
            "scheduler_invalid_spec",
            "external scheduler spec has invalid keys",
            scheduler=spec.get("name") if isinstance(spec.get("name"), str) else None,
            details={"missing": sorted(missing), "unknown": sorted(unknown)},
        )
    name = _validate_scheduler_name(spec["name"])
    raw_command = spec["command"]
    if isinstance(raw_command, (str, bytes)) or not isinstance(raw_command, Sequence):
        raise SchedulerAdapterError(
            "scheduler_invalid_spec",
            "external command must be a non-empty string array",
            scheduler=name,
        )
    command = tuple(
        sys.executable if part == "{python}" else part for part in raw_command
    )
    raw_working_directory = spec.get("working_directory", ".")
    if not isinstance(raw_working_directory, str) or not raw_working_directory:
        raise SchedulerAdapterError(
            "scheduler_invalid_spec",
            "working_directory must be a non-empty string",
            scheduler=name,
        )
    working_directory = Path(raw_working_directory)
    if not working_directory.is_absolute():
        working_directory = context.config_dir / working_directory
    return ExternalProcessScheduler(
        name=name,
        command=command,
        timeout_seconds=spec.get("timeout_seconds", 30.0),
        working_directory=working_directory,
        max_output_bytes=spec.get("max_output_bytes", 4 * 1024 * 1024),
    )


def build_scheduler_runners(
    learned_policy: Any,
    random_seed: int,
    scheduler_specs: Sequence[str | Mapping[str, Any]] | None = None,
    config_dir: str | Path | None = None,
    registry: SchedulerRegistry | None = None,
) -> tuple[SchedulerRunner, ...]:
    """Build a unique ordered runner set from evaluation scheduler specs."""

    selected: Sequence[str | Mapping[str, Any]]
    if scheduler_specs is None:
        selected = DEFAULT_SCHEDULER_NAMES
    elif isinstance(scheduler_specs, (str, bytes)) or not isinstance(
        scheduler_specs, Sequence
    ):
        raise SchedulerAdapterError(
            "scheduler_invalid_spec",
            "evaluation.schedulers must be a non-empty array",
        )
    else:
        selected = scheduler_specs
    if not selected:
        raise SchedulerAdapterError(
            "scheduler_invalid_spec",
            "evaluation.schedulers must be a non-empty array",
        )
    base_dir = Path(config_dir or Path.cwd()).resolve()
    context = SchedulerContext(
        learned_policy=learned_policy,
        random_seed=int(random_seed),
        config_dir=base_dir,
    )
    active_registry = registry or DEFAULT_SCHEDULER_REGISTRY
    runners: list[SchedulerRunner] = []
    names: set[str] = set()
    for index, raw_spec in enumerate(selected):
        if isinstance(raw_spec, str):
            runner = active_registry.create(raw_spec, context)
        elif isinstance(raw_spec, Mapping):
            adapter_type = raw_spec.get("type", "builtin")
            if adapter_type == "builtin":
                unknown = set(raw_spec) - {"type", "name"}
                if unknown or "name" not in raw_spec:
                    raise SchedulerAdapterError(
                        "scheduler_invalid_spec",
                        "builtin scheduler spec requires only type and name",
                        details={
                            "index": index,
                            "unknown": sorted(unknown),
                            "missing_name": "name" not in raw_spec,
                        },
                    )
                runner = active_registry.create(raw_spec["name"], context)
            elif adapter_type == "external":
                runner = _external_runner_from_spec(raw_spec, context)
            else:
                raise SchedulerAdapterError(
                    "scheduler_invalid_spec",
                    "scheduler type must be builtin or external",
                    details={"index": index, "type": adapter_type},
                )
        else:
            raise SchedulerAdapterError(
                "scheduler_invalid_spec",
                "each evaluation scheduler must be a name or object",
                details={"index": index},
            )
        if runner.name in names:
            raise SchedulerAdapterError(
                "scheduler_duplicate",
                f"scheduler {runner.name!r} appears more than once",
                scheduler=runner.name,
            )
        names.add(runner.name)
        runners.append(runner)

    missing_required = {"heft", "masked_mlp"} - names
    if missing_required:
        raise SchedulerAdapterError(
            "scheduler_invalid_spec",
            "evaluation requires heft and masked_mlp",
            details={"missing": sorted(missing_required)},
        )
    return tuple(runners)
