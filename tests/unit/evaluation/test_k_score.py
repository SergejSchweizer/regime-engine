from __future__ import annotations

from dataclasses import replace
from math import inf, isnan, tanh

import pytest

from market_regime_engine.evaluations.k_score import (
    K_SCORE_VERSION,
    KScoreCandidate,
    KScoreFoldEvidence,
    rank_k_candidates,
    score_k_candidate,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
FOLDS = ("outer_001", "outer_002", "outer_003", "outer_004")


def fold(
    fold_id: str,
    *,
    target: float = 0.4,
    baseline: float = 0.0,
    calibration_error: float = 0.1,
    stability: float = 0.8,
    support: float = 0.9,
) -> KScoreFoldEvidence:
    return KScoreFoldEvidence(
        fold_id=fold_id,
        valid=True,
        target_log_score=target,
        baseline_target_log_score=baseline,
        calibration_error=calibration_error,
        stability_score=stability,
        support_score=support,
    )


def candidate(
    state_count: int,
    *,
    feature_hash: str = HASH_A,
    folds: tuple[KScoreFoldEvidence, ...] | None = None,
    target_metric_id: str = "next_return_log_score",
) -> KScoreCandidate:
    return KScoreCandidate(
        state_count=state_count,
        feature_order_hash=feature_hash,
        source_build_id="source-1",
        evaluation_plan_hash=HASH_B,
        target_metric_id=target_metric_id,
        target_horizon="one_step",
        folds=folds or tuple(fold(fold_id) for fold_id in FOLDS),
    )


def test_score_uses_common_target_and_exposes_auditable_components() -> None:
    result = score_k_candidate(candidate(2), latest_fold_id="outer_004")

    assert result.eligible is True
    assert result.valid_fold_rate == 1.0
    assert result.forecast_score == pytest.approx(0.5 + 0.5 * tanh(0.4))
    assert result.calibration_score == pytest.approx(0.9)
    assert result.stability_score == pytest.approx(0.8)
    assert result.mean_support_score == pytest.approx(0.9)
    assert result.total_score is not None


def test_score_formula_binds_every_component_and_complexity_penalty() -> None:
    evidence = (
        fold("outer_001", target=0.0, calibration_error=0.0, stability=0.2, support=0.4),
        fold(
            "outer_002",
            target=1.0,
            baseline=0.25,
            calibration_error=0.25,
            stability=0.4,
            support=0.6,
        ),
        fold(
            "outer_003",
            target=-0.5,
            baseline=0.5,
            calibration_error=0.5,
            stability=0.6,
            support=0.8,
        ),
        fold("outer_004", target=2.0, calibration_error=0.75, stability=0.8, support=1.0),
    )
    result = score_k_candidate(candidate(5, folds=evidence), latest_fold_id="outer_004")

    forecast_values = tuple(
        0.5 + 0.5 * tanh(item.target_log_score - item.baseline_target_log_score)  # type: ignore[operator]
        for item in evidence
    )
    expected_forecast = sum(forecast_values) / len(forecast_values)
    expected_calibration = (1.0 + 0.75 + 0.5 + 0.25) / 4.0
    expected_stability = (0.2 + 0.4 + 0.6 + 0.8) / 4.0
    expected_support = (0.4 + 0.6 + 0.8 + 1.0) / 4.0
    expected_worst = min(forecast_values)
    expected_robustness = 0.50 + 0.25 * expected_worst + 0.25 * expected_support
    expected_total = (
        0.50 * expected_forecast
        + 0.20 * expected_calibration
        + 0.20 * expected_stability
        + 0.10 * expected_robustness
        - 0.03
    )

    assert result.forecast_score == pytest.approx(expected_forecast)
    assert result.calibration_score == pytest.approx(expected_calibration)
    assert result.stability_score == pytest.approx(expected_stability)
    assert result.mean_support_score == pytest.approx(expected_support)
    assert result.worst_fold_forecast_score == pytest.approx(expected_worst)
    assert result.robustness_score == pytest.approx(expected_robustness)
    assert result.complexity_penalty == pytest.approx(0.03)
    assert result.total_score == pytest.approx(expected_total)


@pytest.mark.parametrize(
    ("valid_ids", "eligible", "required_reasons"),
    (
        ({*FOLDS, "outer_005"}, True, ()),
        ({"outer_001", "outer_002", "outer_005"}, False, ("valid-fold rate below 0.80",)),
        ({"outer_001", "outer_002", "outer_003", "outer_005"}, True, ()),
        (
            {"outer_001", "outer_005"},
            False,
            ("valid-fold rate below 0.80", "fewer than three valid outer folds"),
        ),
        (
            set(),
            False,
            (
                "zero valid outer folds",
                "valid-fold rate below 0.80",
                "fewer than three valid outer folds",
                "latest complete outer fold is invalid",
            ),
        ),
    ),
)
def test_eligibility_gates_are_independent_and_keep_invalid_folds_in_denominator(
    valid_ids: set[str], eligible: bool, required_reasons: tuple[str, ...]
) -> None:
    fold_ids = (*FOLDS, "outer_005")
    evidence = tuple(
        fold(fold_id)
        if fold_id in valid_ids
        else KScoreFoldEvidence(fold_id=fold_id, valid=False, invalid_reason="fit failed")
        for fold_id in fold_ids
    )
    result = score_k_candidate(candidate(2, folds=evidence), latest_fold_id="outer_005")

    assert result.eligible is eligible
    assert result.valid_fold_rate == pytest.approx(len(valid_ids) / len(fold_ids))
    assert all(reason in result.rejection_reasons for reason in required_reasons)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("target_log_score", float("nan")),
        ("baseline_target_log_score", inf),
        ("calibration_error", float("nan")),
        ("stability_score", inf),
        ("support_score", float("nan")),
    ),
)
def test_valid_fold_evidence_rejects_nonfinite_values(field: str, value: float) -> None:
    values = {
        "target_log_score": 0.4,
        "baseline_target_log_score": 0.0,
        "calibration_error": 0.1,
        "stability_score": 0.8,
        "support_score": 0.9,
    }
    values[field] = value
    with pytest.raises(ValueError, match="finite"):
        KScoreFoldEvidence(fold_id="outer_001", valid=True, **values)


def test_canonical_cross_k_evidence_ignores_input_order_but_binds_feature_identity() -> None:
    candidates = (
        candidate(2, feature_hash=HASH_A),
        candidate(3, feature_hash=HASH_B),
        candidate(4, feature_hash="c" * 64),
        candidate(5, feature_hash="d" * 64),
    )
    canonical = rank_k_candidates(candidates, latest_fold_id="outer_004", max_workers=1)
    reordered = rank_k_candidates(
        tuple(replace(item, folds=tuple(reversed(item.folds))) for item in reversed(candidates)),
        latest_fold_id="outer_004",
        max_workers=1,
    )

    assert reordered.canonical_json == canonical.canonical_json
    assert reordered.source_hash == canonical.source_hash

    changed_feature = rank_k_candidates(
        (candidate(2, feature_hash="e" * 64),), latest_fold_id="outer_004", max_workers=1
    )
    assert (
        changed_feature.source_hash
        != rank_k_candidates((candidate(2),), latest_fold_id="outer_004", max_workers=1).source_hash
    )


def test_metric_projection_covers_all_legal_slots_and_omits_ineligible_total() -> None:
    invalid = tuple(
        KScoreFoldEvidence(fold_id=fold_id, valid=False, invalid_reason="unsupported")
        for fold_id in FOLDS
    )
    ranking = rank_k_candidates(
        (
            candidate(2),
            candidate(3, feature_hash=HASH_B),
            candidate(4, feature_hash="c" * 64),
            candidate(5, feature_hash="d" * 64, folds=invalid),
        ),
        latest_fold_id="outer_004",
    )
    keys = {point.key for point in ranking.metric_points(timestamp_ms=123)}

    for state_count in (2, 3, 4):
        assert f"k_score_total_score_k{state_count}" in keys
        assert f"k_score_eligible_k{state_count}" in keys
    assert "k_score_total_score_k5" not in keys
    assert "k_score_eligible_k5" in keys
    assert not isnan(
        next(
            point.value
            for point in ranking.metric_points(timestamp_ms=123)
            if point.key == "k_score_eligible_k5"
        )
    )


def test_cross_k_ranking_allows_different_features_but_requires_common_target_contract() -> None:
    result = rank_k_candidates(
        (
            candidate(2, feature_hash=HASH_A),
            candidate(3, feature_hash=HASH_B),
        ),
        latest_fold_id="outer_004",
    )

    assert result.score_version == K_SCORE_VERSION
    assert result.selected_state_count == 2
    assert result.ranked_state_counts == (2, 3)
    assert result.scores[0].feature_order_hash == HASH_A
    assert result.scores[1].feature_order_hash == HASH_B

    with pytest.raises(ValueError, match="target"):
        rank_k_candidates(
            (candidate(2), candidate(3, target_metric_id="different_target")),
            latest_fold_id="outer_004",
        )


def test_higher_forecast_gain_wins_before_complexity_penalty() -> None:
    stronger = tuple(fold(fold_id, target=0.9) for fold_id in FOLDS)
    result = rank_k_candidates(
        (candidate(2), candidate(3, folds=stronger)),
        latest_fold_id="outer_004",
    )
    assert result.selected_state_count == 3


def test_tie_breaks_worst_fold_then_lower_k() -> None:
    k2_folds = tuple(fold(fold_id, target=0.4) for fold_id in FOLDS)
    k3_folds = tuple(fold(fold_id, target=0.4, stability=0.7) for fold_id in FOLDS)
    result = rank_k_candidates(
        (candidate(2, folds=k2_folds), candidate(3, folds=k3_folds)),
        latest_fold_id="outer_004",
    )
    assert result.selected_state_count == 2


def test_eligibility_requires_rate_three_folds_and_latest_fold() -> None:
    invalid_latest = tuple(
        fold(fold_id)
        if fold_id != "outer_004"
        else KScoreFoldEvidence(fold_id=fold_id, valid=False, invalid_reason="fit failed")
        for fold_id in FOLDS
    )
    result = rank_k_candidates(
        (candidate(2, folds=invalid_latest), candidate(3)),
        latest_fold_id="outer_004",
    )
    k2 = next(score for score in result.scores if score.state_count == 2)
    assert k2.eligible is False
    assert "latest complete outer fold is invalid" in k2.rejection_reasons
    assert result.selected_state_count == 3

    only_two_valid = (
        fold("outer_001"),
        fold("outer_002"),
        KScoreFoldEvidence(fold_id="outer_003", valid=False, invalid_reason="support"),
        KScoreFoldEvidence(fold_id="outer_004", valid=False, invalid_reason="support"),
    )
    rejected = score_k_candidate(candidate(2, folds=only_two_valid), latest_fold_id="outer_004")
    assert rejected.total_score is None
    assert rejected.eligible is False
    assert "fewer than three valid outer folds" in rejected.rejection_reasons


def test_cross_k_rejects_fold_or_baseline_mismatch_and_missing_winner_is_explicit() -> None:
    with pytest.raises(ValueError, match="identical source"):
        rank_k_candidates(
            (candidate(2), replace(candidate(3), evaluation_plan_hash="c" * 64)),
            latest_fold_id="outer_004",
        )

    changed_baseline = tuple(fold(fold_id, baseline=0.2) for fold_id in FOLDS)
    with pytest.raises(ValueError, match="common baseline"):
        rank_k_candidates(
            (candidate(2), candidate(3, folds=changed_baseline)),
            latest_fold_id="outer_004",
        )

    no_winner = rank_k_candidates(
        (
            candidate(
                2,
                folds=tuple(
                    KScoreFoldEvidence(fold_id=fold_id, valid=False, invalid_reason="invalid")
                    for fold_id in FOLDS
                ),
            ),
        ),
        latest_fold_id="outer_004",
    )
    assert no_winner.selected_state_count is None
    assert no_winner.no_winner_reason is not None


def test_invalid_fold_cannot_hide_partial_metrics() -> None:
    with pytest.raises(ValueError, match="partial"):
        KScoreFoldEvidence(
            fold_id="outer_001",
            valid=False,
            target_log_score=0.1,
            invalid_reason="failed",
        )


def test_metric_projection_is_complete_and_uses_registered_keys() -> None:
    ranking = rank_k_candidates(
        (candidate(2), candidate(3, feature_hash=HASH_B)),
        latest_fold_id="outer_004",
    )
    points = ranking.metric_points(timestamp_ms=123, step=7)
    keys = {point.key for point in points}
    assert "k_score_total_score_k2" in keys
    assert "k_score_total_score_k3" in keys
    assert "k_score_eligible_k2" in keys
    assert all(point.timestamp_ms == 123 and point.step == 7 for point in points)

    with pytest.raises(ValueError, match="timestamp"):
        ranking.metric_points(timestamp_ms=-1)


def test_process_parallel_and_serial_ranking_have_identical_canonical_evidence() -> None:
    candidates = (
        candidate(4, feature_hash=HASH_B),
        candidate(2),
        candidate(5, folds=tuple(fold(fold_id, target=0.45) for fold_id in FOLDS)),
        candidate(3, feature_hash=HASH_B),
    )

    serial = rank_k_candidates(candidates, latest_fold_id="outer_004", max_workers=1)
    parallel = rank_k_candidates(candidates, latest_fold_id="outer_004", max_workers=2)

    assert parallel.canonical_json == serial.canonical_json
    assert parallel.source_hash == serial.source_hash


def test_score_evidence_rejects_boolean_numeric_fields() -> None:
    with pytest.raises(ValueError, match="boolean"):
        KScoreFoldEvidence(
            fold_id="outer_001",
            valid=1,  # type: ignore[arg-type]
            target_log_score=0.4,
            baseline_target_log_score=0.0,
            calibration_error=0.1,
            stability_score=0.8,
            support_score=0.9,
        )
