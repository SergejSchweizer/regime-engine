from __future__ import annotations

import pytest

from market_regime_engine.feature_discovery.feature_subset_score import (
    FEATURE_SUBSET_SCORE_VERSION,
    FeatureSubsetCandidate,
    FeatureSubsetFoldEvidence,
    rank_feature_subset_scores,
    score_feature_subset,
)

HASH = "a" * 64


def _fold(name: str, *, latest: bool = False, valid: bool = True) -> FeatureSubsetFoldEvidence:
    if not valid:
        return FeatureSubsetFoldEvidence(name, False, latest=latest, invalid_reason="invalid fit")
    return FeatureSubsetFoldEvidence(
        name,
        True,
        latest=latest,
        target_log_score=-1.0,
        baseline_target_log_score=-1.5,
        calibration_error=0.1,
        stability_score=0.8,
        support_score=0.9,
    )


def _candidate(*folds: FeatureSubsetFoldEvidence) -> FeatureSubsetCandidate:
    return FeatureSubsetCandidate(("a", "b"), HASH, "build", "b" * 64, tuple(folds))


def test_score_requires_inner_folds_and_has_no_complexity_penalty() -> None:
    result = score_feature_subset(
        _candidate(_fold("m1"), _fold("m2"), _fold("m3"), _fold("m4", latest=True)),
        latest_fold_id="m4",
    )
    assert result.score_version == FEATURE_SUBSET_SCORE_VERSION
    assert result.eligible is True
    assert result.total_score is not None
    assert result.total_score == pytest.approx(
        0.50 * result.forecast_score
        + 0.20 * result.calibration_score
        + 0.20 * result.stability_score
        + 0.10 * result.robustness_score
    )


def test_score_applies_all_inner_fold_gates() -> None:
    result = score_feature_subset(
        _candidate(_fold("m1"), _fold("m2"), _fold("m3", valid=False), _fold("m4", latest=True)),
        latest_fold_id="m4",
    )
    assert result.eligible is False
    assert "valid-fold rate below 0.80" in result.rejection_reasons
    assert result.total_score is None


def test_outer_test_access_and_wrong_plan_fail_closed() -> None:
    folds = (_fold("m1"), _fold("m2"), _fold("m3"), _fold("m4", latest=True))
    with pytest.raises(ValueError, match="Outer TEST"):
        FeatureSubsetCandidate(("a",), HASH, "build", "b" * 64, folds, outer_test_accessed=True)
    with pytest.raises(ValueError, match="monthly inner"):
        FeatureSubsetCandidate(
            ("a",), HASH, "build", "b" * 64, folds, fold_plan_id="outer_test.v1"
        )


def test_ranking_uses_canonical_tie_breaks_and_feature_tuple() -> None:
    candidates = [
        _candidate(_fold("m1"), _fold("m2"), _fold("m3"), _fold("m4", latest=True)),
        FeatureSubsetCandidate(
            ("a",), HASH, "build", "b" * 64,
            (_fold("m1"), _fold("m2"), _fold("m3"), _fold("m4", latest=True)),
        ),
    ]
    scores = [score_feature_subset(item, latest_fold_id="m4") for item in candidates]
    assert tuple(item.feature_names for item in rank_feature_subset_scores(scores)) == (
        ("a",),
        ("a", "b"),
    )
