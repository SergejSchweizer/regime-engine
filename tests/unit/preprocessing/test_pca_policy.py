from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from market_regime_engine.preprocessing import (
    PCAFitClock,
    PCAFitResult,
    fit_pca_inner_train,
)

START = datetime(2026, 1, 1, tzinfo=UTC)


def _source(count: int = 120) -> tuple[tuple[datetime, ...], np.ndarray]:
    index = np.arange(count, dtype=np.float64)
    return (
        tuple(START + timedelta(days=int(value)) for value in index),
        np.column_stack((index, 2.0 * index + 1.0, np.sin(index / 4.0))),
    )


def test_inner_train_clock_ignores_future_rows_and_counts_incomplete_rows() -> None:
    timestamps, rows = _source()
    rows[45, 1] = np.nan
    clock = PCAFitClock("inner_fold_001", START + timedelta(days=10), START + timedelta(days=89))
    result = fit_pca_inner_train(
        timestamps,
        rows,
        feature_order=("a", "b", "c"),
        inner_fold_id=clock.inner_fold_id,
        fit_start=clock.fit_start,
        fit_end=clock.fit_end,
    )
    future_rows = np.vstack((rows, np.full((20, 3), 1_000_000.0)))
    future_timestamps = timestamps + tuple(
        START + timedelta(days=index) for index in range(120, 140)
    )
    resumed = fit_pca_inner_train(
        future_timestamps,
        future_rows,
        feature_order=("a", "b", "c"),
        inner_fold_id=clock.inner_fold_id,
        fit_start=clock.fit_start,
        fit_end=clock.fit_end,
    )

    assert result.fit_hash == resumed.fit_hash
    assert result.selected_row_count == 79
    assert result.skipped_incomplete_row_count == 1
    assert result.selected_timestamps[0] == START + timedelta(days=10)
    assert result.selected_timestamps[-1] == START + timedelta(days=89)


def test_pca_fit_result_round_trip_is_exact() -> None:
    timestamps, rows = _source()
    result = fit_pca_inner_train(
        timestamps,
        rows,
        feature_order=("a", "b", "c"),
        inner_fold_id="inner_fold_001",
        fit_start=START + timedelta(days=5),
        fit_end=START + timedelta(days=100),
    )

    restored = PCAFitResult.from_canonical_json(result.to_canonical_json())

    assert restored == result
    assert restored.fit_hash == result.fit_hash


def test_pca_fit_clock_rejects_bad_scope_and_source() -> None:
    with pytest.raises(ValueError, match="fit_start"):
        PCAFitClock("inner_fold_001", START + timedelta(days=2), START)
    timestamps, rows = _source()
    with pytest.raises(ValueError, match="strictly increasing"):
        fit_pca_inner_train(
            (timestamps[1], timestamps[0], *timestamps[2:]),
            rows,
            feature_order=("a", "b", "c"),
            inner_fold_id="inner_fold_001",
            fit_start=START,
            fit_end=START + timedelta(days=10),
        )
    with pytest.raises(ValueError, match="no complete TRAIN"):
        fit_pca_inner_train(
            timestamps[:2],
            np.array([[np.nan, 1.0, 2.0], [np.nan, 2.0, 3.0]]),
            feature_order=("a", "b", "c"),
            inner_fold_id="inner_fold_001",
            fit_start=START,
            fit_end=START + timedelta(days=1),
        )
