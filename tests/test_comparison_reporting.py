from __future__ import annotations

import numpy as np

from scripts.compare_task_gnn_mlp import _outcome, _paired_statistics


def test_paired_statistics_are_deterministic_and_respect_direction() -> None:
    mlp = np.asarray(
        [
            [1.00, 0.95, 1.10, 0.90],
            [0.98, 1.02, 1.08, 0.92],
            [1.01, 0.97, 1.04, 0.94],
        ],
        dtype=np.float64,
    )
    task_gnn = mlp - 0.05
    first = _paired_statistics(
        mlp,
        task_gnn,
        bootstrap_samples=2000,
        bootstrap_seed=17,
    )
    second = _paired_statistics(
        mlp,
        task_gnn,
        bootstrap_samples=2000,
        bootstrap_seed=17,
    )

    assert first == second
    assert np.isclose(first["mean_paired_delta"], -0.05)
    assert first["all_seed_scenario_pairs"] == {
        "task_gnn_win": 12,
        "tie": 0,
        "mlp_win": 0,
    }
    assert (
        first["hierarchical_paired_bootstrap"]["excludes_zero_in_task_gnn_direction"]
        is True
    )
    assert _outcome(-2e-9) == "task_gnn_win"
    assert _outcome(0.5e-9) == "tie"
    assert _outcome(2e-9) == "mlp_win"
