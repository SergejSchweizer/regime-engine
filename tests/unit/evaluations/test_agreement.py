from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from market_regime_engine.evaluations.agreement import compare_univariate_to_multivariate


def _evaluation(
    states: tuple[int, ...],
    *,
    state_count: int = 2,
    valid_fold_ids: tuple[str, ...] = ("fold_001", "fold_002", "fold_003", "fold_004", "fold_005"),
    timestamps: tuple[datetime, ...] | None = None,
    source_build_id: str = "build-1",
    evaluation_plan_hash: str = "a" * 64,
    feature_order: tuple[str, ...] = ("feature",),
):
    if timestamps is None:
        start = datetime(2026, 1, 1, tzinfo=UTC)
        timestamps = tuple(start + timedelta(days=index) for index in range(len(states)))
    probabilities = tuple(
        tuple(1.0 if index == state else 0.0 for index in range(state_count)) for state in states
    )
    folds = tuple(
        SimpleNamespace(
            fold_id=fold_id,
            oos_timestamps=timestamps,
            oos_filtered_probabilities=probabilities,
        )
        for fold_id in valid_fold_ids
    )
    return SimpleNamespace(
        source_build_id=source_build_id,
        evaluation_plan_hash=evaluation_plan_hash,
        feature_order=feature_order,
        state_count=state_count,
        valid_folds=folds,
        folds=tuple(SimpleNamespace(fold_id=f"fold_{index:03d}") for index in range(1, 6)),
    )


def test_agreement_is_perfect_for_state_relabeling() -> None:
    result = compare_univariate_to_multivariate(
        "feature", _evaluation((0, 1, 0, 1)), _evaluation((1, 0, 1, 0))
    )
    assert result.shared_fold_ids == ("fold_001", "fold_002", "fold_003", "fold_004", "fold_005")
    assert result.shared_fold_count == 5
    assert result.shared_fold_rate == 1.0
    assert result.shared_timestamp_count == 20
    assert result.dominant_state_nmi == 1.0
    assert result.permutation_hard_agreement == 1.0
    assert result.permutation_mapping == (1, 0)
    assert result.unavailable_reason is None


def test_agreement_reports_nmi_for_independent_labels() -> None:
    result = compare_univariate_to_multivariate(
        "feature", _evaluation((0, 0, 1, 1)), _evaluation((0, 1, 0, 1))
    )
    assert result.dominant_state_nmi == pytest.approx(0.0, abs=1e-12)
    assert result.permutation_hard_agreement == 0.5
    assert result.permutation_mapping == (0, 1)


def test_agreement_returns_no_permutation_for_unequal_state_spaces() -> None:
    result = compare_univariate_to_multivariate(
        "feature", _evaluation((0, 1)), _evaluation((0, 1), state_count=3)
    )
    assert result.dominant_state_nmi == 1.0
    assert result.permutation_hard_agreement is None
    assert result.permutation_mapping is None


def test_agreement_returns_exact_nmi_for_constant_sequences() -> None:
    result = compare_univariate_to_multivariate(
        "feature", _evaluation((0, 0, 0)), _evaluation((1, 1, 1))
    )
    assert result.dominant_state_nmi == 1.0


def test_agreement_returns_unavailable_for_insufficient_support_or_timestamps() -> None:
    insufficient = compare_univariate_to_multivariate(
        "feature",
        _evaluation((0,), valid_fold_ids=("fold_001", "fold_002", "fold_003")),
        _evaluation((0,), valid_fold_ids=("fold_001", "fold_002", "fold_003")),
    )
    assert insufficient.unavailable_reason == "shared valid-fold support below 0.80"
    assert insufficient.dominant_state_nmi is None

    no_common_timestamps = compare_univariate_to_multivariate(
        "feature",
        _evaluation((0,)),
        _evaluation((0,), timestamps=(datetime(2026, 2, 1, tzinfo=UTC),)),
    )
    assert no_common_timestamps.unavailable_reason == "zero shared OOS timestamps"
    assert no_common_timestamps.shared_timestamp_count == 0


def test_agreement_rejects_incompatible_lineage_or_non_univariate_input() -> None:
    univariate = _evaluation((0, 1))
    with pytest.raises(ValueError, match="source build"):
        compare_univariate_to_multivariate(
            "feature", univariate, _evaluation((0, 1), source_build_id="build-2")
        )
    with pytest.raises(ValueError, match="univariate"):
        compare_univariate_to_multivariate("other", univariate, _evaluation((0, 1)))
