from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import duckdb
import pytest

from market_regime_engine.evaluations.task_frontier import SharedTaskFrontier
from market_regime_engine.feature_discovery.ablation import (
    HMMSubsetEvaluation,
    HMMSubsetEvaluator,
    run_one_feature_hmm_ablation,
)
from market_regime_engine.feature_discovery.metadata_store import (
    FeatureSelectionMetadataStore,
    apply_ablation_to_feature_stats,
)
from market_regime_engine.feature_discovery.sffs import FeatureSubsetScore
from tests.unit.feature_discovery.test_metadata_store import make_bundle

SELECTOR_HASH = "a" * 64
PLAN_HASH = "b" * 64
SEED_HASH = "c" * 64


def _evaluation(features: tuple[str, ...]) -> HMMSubsetEvaluation:
    values = {
        ("a", "b", "c"): 3.0,
        ("b", "c"): 2.0,
        ("a", "c"): 3.0,
        ("a", "b"): 4.0,
    }
    return HMMSubsetEvaluation(
        FeatureSubsetScore(features, values[features]),
        "gaussian_hmm",
        2,
        SELECTOR_HASH,
        sha256("|".join(features).encode()).hexdigest(),
        PLAN_HASH,
        SEED_HASH,
    )


def test_ablation_has_one_fresh_fit_per_removal_and_preserves_signed_losses() -> None:
    result = run_one_feature_hmm_ablation(
        ("a", "b", "c"),
        _evaluation,
        selector_contract_hash=SELECTOR_HASH,
        max_workers=8,
    )

    assert tuple(item.removed_feature for item in result.one_feature_results) == ("a", "b", "c")
    assert result.ablation_losses == (1.0, 0.0, -1.0)
    assert len(result.fit_execution_hashes) == 4
    assert result.selected_features == ("a", "b", "c")


def test_invalid_ablation_is_explicit_and_has_no_fabricated_score_or_loss() -> None:
    def evaluate(features: tuple[str, ...]) -> HMMSubsetEvaluation:
        if features == ("a", "c"):
            return HMMSubsetEvaluation(
                None,
                "gaussian_hmm",
                2,
                SELECTOR_HASH,
                "e" * 64,
                PLAN_HASH,
                SEED_HASH,
                "invalid covariance fit",
            )
        return _evaluation(features)

    result = run_one_feature_hmm_ablation(
        ("a", "b", "c"), evaluate, selector_contract_hash=SELECTOR_HASH, max_workers=1
    )
    invalid = result.one_feature_results[1]
    assert invalid.score is None
    assert invalid.ablation_loss is None
    assert invalid.hmm_evaluation.invalid_reason == "invalid covariance fit"


def test_ablation_rejects_changed_inner_plan_or_seed_identity() -> None:
    def changed(features: tuple[str, ...]) -> HMMSubsetEvaluation:
        result = _evaluation(features)
        return result if features == ("a", "b", "c") else replace(result, seed_identity="f" * 64)

    with pytest.raises(ValueError, match="seed_identity"):
        run_one_feature_hmm_ablation(
            ("a", "b", "c"), changed, selector_contract_hash=SELECTOR_HASH, max_workers=1
        )


def test_ablation_can_reuse_a_caller_owned_shared_frontier() -> None:
    serial = run_one_feature_hmm_ablation(
        ("a", "b", "c"),
        _evaluation,
        selector_contract_hash=SELECTOR_HASH,
        max_workers=1,
    )
    with SharedTaskFrontier[HMMSubsetEvaluator, HMMSubsetEvaluation | None](
        max_workers=2
    ) as frontier:
        result = run_one_feature_hmm_ablation(
            ("a", "b", "c"),
            _evaluation,
            selector_contract_hash=SELECTOR_HASH,
            frontier=frontier,
        )
    assert result.ablation_losses == (1.0, 0.0, -1.0)
    assert result == serial


def test_ablation_losses_are_persisted_without_clipping(tmp_path: Path) -> None:
    result = run_one_feature_hmm_ablation(
        ("a", "b", "c"), _evaluation, selector_contract_hash=SELECTOR_HASH, max_workers=1
    )
    original = make_bundle()
    rows = tuple(
        replace(original.fold_feature_stats[0], feature_name=feature) for feature in ("a", "b", "c")
    )
    # The adapter is intentionally independent of the source quality values.
    adapted = apply_ablation_to_feature_stats(
        rows,
        result,
    )
    store = FeatureSelectionMetadataStore(tmp_path)
    assert store.commit_fold_feature_stats(adapted) is True
    with duckdb.connect(str(store.database), read_only=True) as connection:
        values = connection.execute(
            "SELECT feature_name, ablation_loss FROM fold_feature_stats ORDER BY feature_name"
        ).fetchall()
    assert values == [("a", 1.0), ("b", 0.0), ("c", -1.0)]
