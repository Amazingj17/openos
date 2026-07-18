from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREREGISTER = ROOT / "configs" / "p1_a05_size_robustness_preregister.json"
BASE_CONFIG = ROOT / "configs" / "stg_ppo_5seed.json"
EVALUATION_CONTRACT = ROOT / "configs" / "p1_b02_evaluation_contract.json"
ANALYSIS_SCRIPT = ROOT / "scripts" / "analyze_p1_a05_size_robustness.py"


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_p1_a05_freezes_one_transition_budget_matched_intervention() -> None:
    design = load_json(PREREGISTER)
    base = load_json(BASE_CONFIG)
    intervention = design["single_intervention"]
    assert isinstance(intervention, dict)
    assert intervention["only_changed_component"] == "PPO rollout scenario source plan"

    plan = intervention["ppo_rollout_plan"]
    assert isinstance(plan, list)
    assert [item["epoch"] for item in plan] == [1, 2]
    for item in plan:
        assert item["episode_count"] == 90
        assert item["transition_count"] == 6000
        assert item["stg_train_split_indices"]["transition_count"] == 3000
        assert item["synthetic"]["transition_count"] == 3000
        assert (
            item["stg_train_split_indices"]["scenario_count"]
            * item["stg_train_split_indices"]["task_count"]
            == 3000
        )
        assert (
            item["synthetic"]["scenario_count"] * item["synthetic"]["task_count"]
            == 3000
        )

    stg_indices = {
        index
        for item in plan
        for index in range(
            item["stg_train_split_indices"]["start_inclusive"],
            item["stg_train_split_indices"]["end_inclusive"] + 1,
        )
    }
    synthetic_seeds = {
        seed
        for item in plan
        for seed in range(
            item["synthetic"]["seed_start_inclusive"],
            item["synthetic"]["seed_end_inclusive"] + 1,
        )
    }
    assert stg_indices == set(range(120))
    assert synthetic_seeds == set(range(20261001, 20261061))

    budget = intervention["budget"]
    assert budget["training_seeds"] == base["seeds"]
    assert budget["ppo_epochs_per_seed"] == base["ppo"]["epochs"] == 2
    assert budget["transitions_per_epoch_per_seed"] == 6000
    assert budget["formal_total_transitions"] == 60000


def test_p1_a05_preserves_frozen_algorithm_and_id_selection() -> None:
    design = load_json(PREREGISTER)
    base = load_json(BASE_CONFIG)
    frozen = design["frozen_invariants"]
    assert isinstance(frozen, dict)
    ppo = base["ppo"]
    assert isinstance(ppo, dict)
    for name in (
        "gamma",
        "gae_lambda",
        "clip_ratio",
        "entropy_coefficient",
        "target_kl",
        "actor_learning_rate",
        "value_learning_rate",
        "value_hidden_dim",
        "update_epochs",
        "minibatch_size",
        "gradient_clip",
    ):
        assert frozen[name] == ppo[name]
    assert frozen["hidden_dim"] == base["behavior_cloning"]["hidden_dim"]
    assert frozen["excluded_features"] == base["features"]["exclude"]
    assert frozen["checkpoint_selection"]["metric"] == base["selection"]["metric"]
    assert frozen["checkpoint_selection"]["split"].startswith("frozen 30-scenario ID")


def test_p1_a05_training_seeds_are_disjoint_from_frozen_size_ood() -> None:
    design = load_json(PREREGISTER)
    contract = load_json(EVALUATION_CONTRACT)
    plan = design["single_intervention"]["ppo_rollout_plan"]
    training_seeds = {
        seed
        for item in plan
        for seed in range(
            item["synthetic"]["seed_start_inclusive"],
            item["synthetic"]["seed_end_inclusive"] + 1,
        )
    }
    size_slice = next(item for item in contract["slices"] if item["id"] == "ood_size")
    development_seeds = set(size_slice["source"]["generator_seeds"])
    assert training_seeds.isdisjoint(development_seeds)
    assert "public_test" not in contract["modes"]["development"]
    assert design["formal_execution"]["public_test"] == "forbidden"
    assert any(
        "Do not start training before member B signs" in item
        for item in design["prohibitions"]
    )


def test_p1_a05_portable_source_bindings_match_tracked_inputs() -> None:
    design = load_json(PREREGISTER)
    bindings = design["evidence_bindings"]
    assert bindings["base_training_config"]["canonical_sha256"] == (
        canonical_json_sha256(load_json(BASE_CONFIG))
    )
    assert bindings["evaluation_contract"]["canonical_sha256"] == (
        canonical_json_sha256(load_json(EVALUATION_CONTRACT))
    )
    normalized_script = (
        ANALYSIS_SCRIPT.read_text(encoding="utf-8")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .encode("utf-8")
    )
    assert bindings["root_cause_script"]["normalized_lf_sha256"] == (
        hashlib.sha256(normalized_script).hexdigest()
    )
