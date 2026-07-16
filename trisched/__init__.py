"""TriSched minimum executable scheduling framework."""

from .scenario import (
    Scenario,
    ScenarioValidationError,
    generate_dataset,
    generate_scenario,
)

__all__ = [
    "Scenario",
    "ScenarioValidationError",
    "generate_dataset",
    "generate_scenario",
]
__version__ = "0.1.0"
