"""Pure global absolute-Spearman distance for one quality-filtered TRAIN snapshot."""

from __future__ import annotations

from datetime import datetime
from math import fsum, isfinite, sqrt

import numpy as np
from scipy.stats import rankdata  # type: ignore[import-untyped]

from market_regime_engine.feature_discovery.contracts import (
    MIN_PAIRWISE_OBSERVATIONS,
    RHO_CLIP_TOLERANCE,
    DiscoveryStatus,
    DistanceMatrixResult,
    QualityFilterResult,
)
from market_regime_engine.features.ports import FeatureSnapshot


def _average_ranks(values: tuple[float, ...]) -> tuple[float, ...]:
    """Return one-based average ranks, preserving the original observation order."""

    if not values:
        raise ValueError("rank input cannot be empty")
    if any(not isfinite(value) for value in values):
        raise ValueError("rank input must contain finite values")
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[ordered[position][0]] = average_rank
        start = end
    return tuple(ranks)


def _clip_correlation(correlation: float) -> float:
    if not isfinite(correlation):
        raise ValueError("Spearman correlation must be finite")
    if correlation < -1.0 - RHO_CLIP_TOLERANCE or correlation > 1.0 + RHO_CLIP_TOLERANCE:
        raise ValueError("Spearman correlation exceeds [-1,1] beyond tolerance")
    return min(1.0, max(-1.0, correlation))


def _rank_pearson_correlation(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman pair requires equal inputs with at least two observations")
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = fsum(left_ranks) / len(left_ranks)
    right_mean = fsum(right_ranks) / len(right_ranks)
    left_centered = tuple(value - left_mean for value in left_ranks)
    right_centered = tuple(value - right_mean for value in right_ranks)
    left_ss = fsum(value * value for value in left_centered)
    right_ss = fsum(value * value for value in right_centered)
    if left_ss <= 0.0 or right_ss <= 0.0:
        raise ValueError("Spearman correlation is undefined for a constant feature pair")
    correlation = fsum(
        left_value * right_value
        for left_value, right_value in zip(left_centered, right_centered, strict=True)
    ) / sqrt(left_ss * right_ss)
    return _clip_correlation(correlation)


def _rank_pearson_correlation_array(
    left: np.ndarray,
    right: np.ndarray,
    *,
    left_ranks: np.ndarray | None = None,
    right_ranks: np.ndarray | None = None,
) -> float:
    """Compute one pair in native array code after pairwise support filtering."""

    if left.size != right.size or left.size < 2:
        raise ValueError("Spearman pair requires equal inputs with at least two observations")
    resolved_left_ranks = rankdata(left, method="average") if left_ranks is None else left_ranks
    resolved_right_ranks = rankdata(right, method="average") if right_ranks is None else right_ranks
    left_centered = resolved_left_ranks - np.mean(resolved_left_ranks)
    right_centered = resolved_right_ranks - np.mean(resolved_right_ranks)
    left_ss = float(np.dot(left_centered, left_centered))
    right_ss = float(np.dot(right_centered, right_centered))
    if left_ss <= 0.0 or right_ss <= 0.0:
        raise ValueError("Spearman correlation is undefined for a constant feature pair")
    correlation = float(np.dot(left_centered, right_centered) / sqrt(left_ss * right_ss))
    return _clip_correlation(correlation)


def _validate_snapshot(
    snapshot: FeatureSnapshot, quality: QualityFilterResult
) -> tuple[tuple[float | None, ...], ...]:
    if not isinstance(snapshot, FeatureSnapshot):
        raise TypeError("distance calculation requires a FeatureSnapshot")
    if not isinstance(quality, QualityFilterResult):
        raise TypeError("distance calculation requires a QualityFilterResult")
    if quality.status is not DiscoveryStatus.VALID:
        raise ValueError("distance calculation requires valid quality evidence")
    quality_order = tuple(item.feature_name for item in quality.features)
    quality_positions = tuple(item.source_position for item in quality.features)
    if snapshot.feature_names != quality_order:
        raise ValueError("feature snapshot columns must match quality catalog order")
    if quality_positions != tuple(sorted(quality_positions)) or len(set(quality_positions)) != len(
        quality_positions
    ):
        raise ValueError("quality feature positions must be unique and canonical")
    if snapshot.lineage.source_build_id != quality.source_build_id:
        raise ValueError("feature snapshot and quality evidence build IDs must match")
    if len(snapshot.rows) != quality.train_source_observation_count:
        raise ValueError("quality denominator must equal supplied TRAIN row count")
    if snapshot.skipped_incomplete_row_count:
        raise ValueError("distance calculation requires all source rows")
    if not snapshot.rows:
        raise ValueError("distance calculation requires non-empty TRAIN rows")

    expected_dimension = len(snapshot.feature_names)
    copied_rows: list[tuple[float | None, ...]] = []
    previous: datetime | None = None
    for row in snapshot.rows:
        if previous is not None and row.timestamp <= previous:
            raise ValueError("distance rows must be strictly increasing")
        if len(row.values) != expected_dimension:
            raise ValueError("distance row dimension does not match feature order")
        copied_values: list[float | None] = []
        for value in row.values:
            if value is None:
                copied_values.append(None)
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("non-null distance values must be numeric") from exc
            if not isfinite(numeric):
                raise ValueError("non-null distance values must be finite")
            copied_values.append(numeric)
        copied_rows.append(tuple(copied_values))
        previous = row.timestamp
    return tuple(copied_rows)


def global_absolute_spearman_distance(
    snapshot: FeatureSnapshot,
    quality: QualityFilterResult,
) -> DistanceMatrixResult:
    """Compute the canonical distance matrix for every eligible feature.

    Rows are complete only for the particular pair being evaluated.  The
    quality-filtered feature order is retained exactly, and every supplied
    source value is validated before any pair is emitted.
    """

    rows = _validate_snapshot(snapshot, quality)
    all_feature_names = tuple(item.feature_name for item in quality.features)
    eligible_names = quality.eligible_features
    feature_positions = {name: index for index, name in enumerate(all_feature_names)}
    eligible_indices = tuple(feature_positions[name] for name in eligible_names)
    if len(eligible_indices) < 2:
        raise ValueError("distance calculation requires at least two eligible features")

    matrix = np.asarray(
        [
            [np.nan if row[index] is None else row[index] for index in eligible_indices]
            for row in rows
        ],
        dtype=np.float64,
    )
    size = matrix.shape[1]
    missing = np.isnan(matrix).any(axis=0)
    full_ranks = tuple(
        None if missing[index] else rankdata(matrix[:, index], method="average")
        for index in range(size)
    )
    distances = [[0.0 for _ in range(size)] for _ in range(size)]
    supports = [[0 for _ in range(size)] for _ in range(size)]
    correlations = [[0.0 for _ in range(size)] for _ in range(size)]

    for left_index in range(size):
        for right_index in range(left_index, size):
            valid = ~np.isnan(matrix[:, left_index]) & ~np.isnan(matrix[:, right_index])
            support = int(np.count_nonzero(valid))
            supports[left_index][right_index] = support
            supports[right_index][left_index] = support
            if support < MIN_PAIRWISE_OBSERVATIONS:
                raise ValueError(
                    f"pairwise support for {eligible_names[left_index]} and "
                    f"{eligible_names[right_index]} is {support}; "
                    f"requires at least {MIN_PAIRWISE_OBSERVATIONS}"
                )
            if left_index == right_index:
                correlation = 1.0
                distance = 0.0
            else:
                left_values = matrix[valid, left_index]
                right_values = matrix[valid, right_index]
                if not missing[left_index] and not missing[right_index]:
                    correlation = _rank_pearson_correlation_array(
                        left_values,
                        right_values,
                        left_ranks=full_ranks[left_index],
                        right_ranks=full_ranks[right_index],
                    )
                else:
                    correlation = _rank_pearson_correlation_array(left_values, right_values)
                distance = 1.0 - abs(correlation)
                if distance < 0.0 and distance >= -RHO_CLIP_TOLERANCE:
                    distance = 0.0
                if not isfinite(distance) or not 0.0 <= distance <= 1.0:
                    raise ValueError("Spearman distance must be finite in [0,1]")
            distances[left_index][right_index] = distance
            distances[right_index][left_index] = distance
            correlations[left_index][right_index] = correlation
            correlations[right_index][left_index] = correlation

    return DistanceMatrixResult(
        feature_order=eligible_names,
        distances=tuple(tuple(row) for row in distances),
        pairwise_support=tuple(tuple(row) for row in supports),
        minimum_pairwise_observations=MIN_PAIRWISE_OBSERVATIONS,
        spearman_correlations=tuple(tuple(row) for row in correlations),
    )


compute_global_spearman_distance = global_absolute_spearman_distance
compute_distance_matrix = global_absolute_spearman_distance


__all__ = [
    "compute_distance_matrix",
    "compute_global_spearman_distance",
    "global_absolute_spearman_distance",
]
