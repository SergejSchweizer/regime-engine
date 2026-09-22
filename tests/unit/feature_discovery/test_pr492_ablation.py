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
    FoldFeatureStat,
    PcaLoading,
    apply_ablation_to_feature_stats,
    apply_pca_credit_to_feature_stats,
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


def test_pca_credit_uses_squared_loading_and_excludes_direct_features() -> None:
    selected = ("family_pc_vix_1", "direct_feature")

    def evaluate(features: tuple[str, ...]) -> HMMSubsetEvaluation:
        values = {
            selected: 5.0,
            ("direct_feature",): 1.0,
            ("family_pc_vix_1",): 3.0,
        }
        return HMMSubsetEvaluation(
            FeatureSubsetScore(features, values[features]),
            "gaussian_hmm",
            2,
            SELECTOR_HASH,
            sha256("|".join(features).encode()).hexdigest(),
        )

    result = run_one_feature_hmm_ablation(
        selected, evaluate, selector_contract_hash=SELECTOR_HASH, max_workers=1
    )
    profile_hash = "b" * 64
    rows = (
        FoldFeatureStat(
            "fold-001",
            profile_hash,
            "build",
            "family_pc_vix_1",
            True,
            None,
            False,
            True,
            None,
            True,
            True,
            False,
            None,
        ),
        FoldFeatureStat(
            "fold-001",
            profile_hash,
            "build",
            "source_a",
            True,
            None,
            False,
            True,
            None,
            False,
            True,
            False,
            None,
        ),
        FoldFeatureStat(
            "fold-001",
            profile_hash,
            "build",
            "direct_feature",
            True,
            None,
            True,
            False,
            None,
            True,
            True,
            False,
            None,
        ),
    )
    loadings = (
        PcaLoading("fold-001", profile_hash, "build", "vix", 1, "source_a", 0.5, 0.25, 0.8),
    )
    credited = apply_pca_credit_to_feature_stats(
        apply_ablation_to_feature_stats(rows, result), loadings, result
    )

    assert credited[1].pca_credit == 1.0
    assert credited[2].pca_credit is None
    assert credited[2].ablation_loss == 2.0


def test_global_stats_are_cumulative_and_replay_safe(tmp_path: Path) -> None:
    store = FeatureSelectionMetadataStore(tmp_path)
    profile_hash = "b" * 64
    first = (
        FoldFeatureStat(
            "fold-001",
            profile_hash,
            "build",
            "source_a",
            True,
            None,
            False,
            True,
            1.0,
            False,
            True,
            True,
            4.0,
        ),
    )
    second = (
        FoldFeatureStat(
            "fold-002",
            profile_hash,
            "build",
            "source_a",
            True,
            None,
            False,
            True,
            2.0,
            False,
            True,
            False,
            None,
        ),
    )
    assert store.commit_fold_feature_stats(first) is True
    assert store.commit_fold_feature_stats(second) is True
    assert store.commit_fold_feature_stats(second) is True
    with duckdb.connect(str(store.database), read_only=True) as connection:
        row = connection.execute(
            """
            SELECT eligible_folds, quality_pass_folds, selected_folds,
                   selection_rate, mean_ablation_loss, median_ablation_loss,
                   total_pca_credit, mean_pca_credit, last_selected_fold,
                   consecutive_unused_folds
            FROM feature_global_stats WHERE feature_name = 'source_a'
            """
        ).fetchone()
    assert row == (2, 2, 1, 0.5, 4.0, 4.0, 3.0, 1.5, "fold-001", 1)
