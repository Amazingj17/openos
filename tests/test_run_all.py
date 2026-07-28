from __future__ import annotations

import run_all


def test_run_all_forwards_arguments_to_model_comparison(monkeypatch) -> None:
    received: list[list[str] | None] = []

    def fake_main(argv: list[str] | None = None) -> int:
        received.append(argv)
        return 7

    monkeypatch.setattr(run_all, "_prepare_benchmark", lambda: 0)
    monkeypatch.setattr(run_all, "_run_model_comparison", fake_main)

    assert run_all.main(["--resume", "--output", "outputs/test"]) == 7
    assert received == [["--resume", "--output", "outputs/test"]]


def test_run_all_reuses_verified_default_results(monkeypatch) -> None:
    called = False

    def fake_training(argv: list[str] | None = None) -> int:
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(
        run_all,
        "_reuse_completed_default_results",
        lambda: run_all.DEFAULT_COMPARISON_ROOT / "results" / "comparison.html",
    )
    monkeypatch.setattr(run_all, "_run_model_comparison", fake_training)

    assert run_all.main([]) == 0
    assert called is False


def test_run_all_stops_when_benchmark_preparation_fails(monkeypatch) -> None:
    called = False

    def fake_training(argv: list[str] | None = None) -> int:
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(run_all, "_prepare_benchmark", lambda: 2)
    monkeypatch.setattr(run_all, "_reuse_completed_default_results", lambda: None)
    monkeypatch.setattr(run_all, "_run_model_comparison", fake_training)

    assert run_all.main([]) == 2
    assert called is False
