from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

import trisched.p1_a05 as p1_a05
import trisched.ppo as ppo_module
from trisched.bc import BehaviorCloningError
from trisched.cli import main
from trisched.learning import FEATURE_NAMES, TEACHER_FEATURE_NAMES, MaskedMLPPolicy
from trisched.p1_a05 import PREPARED_MANIFEST_NAME, load_p1_a05_config
from trisched.ppo import train_masked_ppo


ROOT = Path(__file__).resolve().parents[1]
TRACKED_CONFIG = ROOT / "configs" / "p1_a05_size_robustness.json"


def _isolated_config(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    payload = json.loads(TRACKED_CONFIG.read_text(encoding="utf-8"))
    payload["preregister"]["path"] = str(
        ROOT / "configs" / "p1_a05_size_robustness_preregister.json"
    )
    payload["development"]["contract"] = str(
        ROOT / "configs" / "p1_b02_evaluation_contract.json"
    )
    payload["output_dir"] = str(tmp_path / "formal-output")
    payload["prepared_input_dir"] = str(tmp_path / "prepared-input")
    payload["implementation_review"] = str(tmp_path / "missing-review.json")
    path = tmp_path / "p1_a05.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path, payload


def _write_variant(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> Path:
    path, payload = _isolated_config(tmp_path)
    mutate(payload)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_tracked_p1_a05_config_loads_and_keeps_the_single_intervention() -> None:
    config = load_p1_a05_config(TRACKED_CONFIG)
    assert config["ppo"]["episodes_per_epoch"] == 90
    assert config["ppo"]["gamma"] == 1.0
    assert config["model"]["policy_class"].endswith("MaskedMLPPolicy")
    assert config["public_test"] == "forbidden"


@pytest.mark.parametrize(
    ("name", "mutate"),
    [
        ("unknown", lambda value: value.__setitem__("unreviewed", True)),
        ("gamma", lambda value: value["ppo"].__setitem__("gamma", 0.99)),
        ("model", lambda value: value["model"].__setitem__("hidden_dim", 64)),
        (
            "episodes",
            lambda value: value["rollout_plan"][0].__setitem__("episode_count", 91),
        ),
        (
            "transitions",
            lambda value: value["rollout_plan"][0].__setitem__(
                "transition_count", 6001
            ),
        ),
        ("public-test", lambda value: value.__setitem__("public_test", "allowed")),
    ],
)
def test_p1_a05_config_rejects_unreviewed_variables(
    tmp_path: Path,
    name: str,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    with pytest.raises(BehaviorCloningError):
        load_p1_a05_config(_write_variant(tmp_path / name, mutate))


@pytest.mark.parametrize(
    "epoch_ids",
    [
        [["train-0", "train-1"]],
        [["train-0"], ["train-1", "train-2"]],
        [["train-0", "train-0"], ["train-1", "train-2"]],
        [["train-0", "unknown"], ["train-1", "train-2"]],
    ],
)
def test_masked_ppo_rejects_invalid_frozen_epoch_scenario_ids(
    monkeypatch: pytest.MonkeyPatch,
    epoch_ids: list[list[str]],
) -> None:
    monkeypatch.setattr(ppo_module, "_manifest_records", lambda *args, **kwargs: {})
    scenarios = [
        SimpleNamespace(id=f"train-{index}", task_count=50) for index in range(3)
    ]
    feature_names = tuple(
        name for name in FEATURE_NAMES if name not in TEACHER_FEATURE_NAMES
    )
    policy = MaskedMLPPolicy(seed=7, feature_names=feature_names)
    with pytest.raises(BehaviorCloningError) as captured:
        train_masked_ppo(
            policy,
            scenarios,
            {},
            [],
            {},
            {"epochs": 2, "episodes_per_epoch": 2},
            seed=7,
            epoch_scenario_ids=epoch_ids,
        )
    assert captured.value.code == "ppo_rollout_plan"


def test_p1_a05_dry_run_counts_exact_frozen_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, _ = _isolated_config(tmp_path)
    config = load_p1_a05_config(config_path)
    root = tmp_path / "prepared-input"
    root.mkdir()
    (root / PREPARED_MANIFEST_NAME).write_text("{}\n", encoding="utf-8")
    stg = [
        SimpleNamespace(id=f"stg-{index:03d}", task_count=50) for index in range(120)
    ]
    synthetic = [
        SimpleNamespace(id=f"synthetic-{index:03d}", task_count=100)
        for index in range(60)
    ]
    monkeypatch.setattr(p1_a05, "load_frozen_split", lambda *args, **kwargs: stg)
    monkeypatch.setattr(
        p1_a05,
        "_load_prepared_synthetic",
        lambda *args, **kwargs: synthetic,
    )

    report = p1_a05._build_dry_run_report(config_path, config, root, {})

    assert [item["episode_count"] for item in report["epochs"]] == [90, 90]
    assert [item["transition_count"] for item in report["epochs"]] == [6000, 6000]
    assert all(
        item["task_count_counts"] == {"50": 60, "100": 30} for item in report["epochs"]
    )
    assert report["checkpoint_loaded"] is False
    assert report["optimizer_created"] is False
    assert report["training_started"] is False
    assert report["public_test_accessed"] is False


def test_cli_refuses_training_before_review_without_creating_output_or_loading_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path, payload = _isolated_config(tmp_path)

    def forbidden_checkpoint_load(*args: object, **kwargs: object) -> None:
        raise AssertionError("checkpoint loading occurred before implementation review")

    monkeypatch.setattr(MaskedMLPPolicy, "load", forbidden_checkpoint_load)
    return_code = main(["train-p1-a05", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert return_code == 2
    assert "p1_a05_review_missing" in captured.err
    assert not Path(payload["output_dir"]).exists()


def test_formal_training_rejects_source_changes_after_the_reviewed_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "formal-output"
    monkeypatch.setattr(
        p1_a05.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stderr="controlled source changed",
        ),
    )

    with pytest.raises(BehaviorCloningError) as captured:
        p1_a05._assert_approved_source_matches(ROOT, "a" * 40)

    assert captured.value.code == "p1_a05_formal_commit"
    assert not output.exists()


def test_formal_source_gate_allows_a_later_receipt_only_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        p1_a05.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stderr=""),
    )

    p1_a05._assert_approved_source_matches(ROOT, "a" * 40)
