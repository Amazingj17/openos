"""Reference external scheduler for the TriSched JSON-over-stdio protocol."""

from __future__ import annotations

import json
import sys
from typing import Any

from trisched.env import run_policy
from trisched.policies import HeftPolicy
from trisched.scenario import Scenario
from trisched.schedulers import EXTERNAL_SCHEDULER_PROTOCOL_VERSION


def _load_request() -> tuple[str, Scenario]:
    payload: Any = json.load(sys.stdin)
    required = {"protocol_version", "scheduler_name", "scenario"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError(f"request keys must be exactly {sorted(required)}")
    if payload["protocol_version"] != EXTERNAL_SCHEDULER_PROTOCOL_VERSION:
        raise ValueError("unsupported protocol_version")
    scheduler_name = payload["scheduler_name"]
    if not isinstance(scheduler_name, str) or not scheduler_name:
        raise ValueError("scheduler_name must be a non-empty string")
    return scheduler_name, Scenario.from_dict(payload["scenario"])


def main() -> int:
    try:
        scheduler_name, scenario = _load_request()
        result = run_policy(scenario, HeftPolicy())
        response = {
            "protocol_version": EXTERNAL_SCHEDULER_PROTOCOL_VERSION,
            "scheduler_name": scheduler_name,
            "scenario_id": scenario.id,
            "makespan": result.makespan,
            "entries": result.to_dict()["entries"],
        }
        print(
            json.dumps(
                response, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            )
        )
    except Exception as error:
        print(f"external scheduler error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
