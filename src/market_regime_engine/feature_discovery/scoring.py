"""Distribution-sensitive raw-feature scores against one causal teacher."""

from __future__ import annotations

from math import fsum, isfinite, log
from typing import Any, cast

import numpy as np

from market_regime_engine.feature_discovery.contracts import (
    FEATURE_SCORE_BIN_COUNT,
    FEATURE_SCORE_TIE_TOLERANCE,
    MIN_FEATURE_SCORE_COVERAGE,
    MIN_FEATURE_SCORE_OBSERVATIONS,
    FeatureRegimeScore,
    ProvisionalTeacherReference,
    QualityFilterResult,
)
from market_regime_engine.features.ports import FeatureSnapshot

_TIMESTAMP_COLUMN = "timestamp_m1"
_PROBABILITY_TOLERANCE = 1.0e-10
_ENTROPY_TOLERANCE = 1.0e-12


def _finite_value(value: object, name: str) -> float:
    try:
        numeric = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _validate_inputs(
    snapshot: FeatureSnapshot,
    quality: QualityFilterResult,
    teacher: ProvisionalTeacherReference,
) -> tuple[dict[str, tuple[float | None, ...]], tuple[int, ...]]:
    if not isinstance(snapshot, FeatureSnapshot):
        raise TypeError("feature scoring requires a FeatureSnapshot")
    if not isinstance(quality, QualityFilterResult):
        raise TypeError("feature scoring requires a QualityFilterResult")
    if not isinstance(teacher, ProvisionalTeacherReference):
        raise TypeError("feature scoring requires a ProvisionalTeacherReference")
    quality_names = tuple(item.feature_name for item in quality.features)
    if snapshot.feature_names != quality_names:
        raise ValueError("feature snapshot columns must match quality evidence order")
    if snapshot.lineage.source_build_id != quality.source_build_id:
        raise ValueError("feature snapshot and quality source builds differ")
    if teacher.source_build_id != quality.source_build_id:
        raise ValueError("teacher and quality source builds differ")
    if not teacher.timestamps:
        raise ValueError("feature scoring requires non-empty teacher timestamps")
    source_timestamps = tuple(row.timestamp for row in snapshot.rows)
    source_index = {timestamp: index for index, timestamp in enumerate(source_timestamps)}
    missing_timestamps = tuple(
        timestamp for timestamp in teacher.timestamps if timestamp not in source_index
    )
    if missing_timestamps:
        raise ValueError("teacher timestamps must be present in the feature snapshot")
    teacher_indices = tuple(source_index[timestamp] for timestamp in teacher.timestamps)
    if teacher_indices != tuple(sorted(teacher_indices)) or len(set(teacher_indices)) != len(
        teacher_indices
    ):
        raise ValueError("teacher timestamps must follow the source snapshot order")

    values_by_feature: dict[str, tuple[float | None, ...]] = {}
    for feature_index, feature_name in enumerate(snapshot.feature_names):
        values: list[float | None] = []
        for row in snapshot.rows:
            value = row.values[feature_index]
            values.append(None if value is None else _finite_value(value, feature_name))
        values_by_feature[feature_name] = tuple(values[index] for index in teacher_indices)
    return values_by_feature, teacher_indices


def _average_ranks(values: tuple[float, ...]) -> tuple[float, ...]:
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and ordered[end][1] == ordered[position][1]:
            end += 1
        average = (position + 1 + end) / 2.0
        for original_index, _ in ordered[position:end]:
            ranks[original_index] = average
        position = end
    return tuple(ranks)


def _teacher_state_masses(
    probabilities: tuple[tuple[float, ...], ...],
) -> tuple[float, ...]:
    count = len(probabilities)
    state_count = len(probabilities[0])
    return tuple(
        fsum(probability[state] for probability in probabilities) / count
        for state in range(state_count)
    )


def _empty_support_score(
    feature_name: str,
    canonical_ordinal: int,
    teacher: ProvisionalTeacherReference,
) -> FeatureRegimeScore:
    state_masses = _teacher_state_masses(teacher.filtered_probabilities)
    state_entropy = -fsum(mass * log(mass) for mass in state_masses if mass > 0.0)
    joint = (state_masses, *((0.0,) * teacher.state_count for _ in range(9)))
    return FeatureRegimeScore(
        feature_name=feature_name,
        canonical_ordinal=canonical_ordinal,
        coverage=0.0,
        observation_count=0,
        bin_counts=(0,) * FEATURE_SCORE_BIN_COUNT,
        joint_bin_state_masses=tuple(joint),
        bin_masses=(1.0,) + (0.0,) * (FEATURE_SCORE_BIN_COUNT - 1),
        state_masses=state_masses,
        mutual_information=0.0,
        state_entropy=state_entropy,
        state_information_ratio=None,
        state_weights=state_masses,
        state_means=(0.0,) * teacher.state_count,
        state_variances=(0.0,) * teacher.state_count,
        between_variance=0.0,
        within_variance=0.0,
        eta_squared=None,
        eligible=False,
        exclusion_reason="zero_teacher_support",
    )


def _score_feature_values(
    feature_name: str,
    canonical_ordinal: int,
    values: tuple[float | None, ...],
    teacher: ProvisionalTeacherReference,
) -> FeatureRegimeScore:
    supported = tuple(
        (value, probability)
        for value, probability in zip(values, teacher.filtered_probabilities, strict=True)
        if value is not None
    )
    observation_count = len(supported)
    coverage = observation_count / len(teacher.timestamps)
    if observation_count == 0:
        return _empty_support_score(feature_name, canonical_ordinal, teacher)

    numeric_values = tuple(value for value, _ in supported)
    probabilities = tuple(probability for _, probability in supported)
    ranks = _average_ranks(numeric_values)
    bins = tuple(
        min(
            FEATURE_SCORE_BIN_COUNT - 1,
            int(FEATURE_SCORE_BIN_COUNT * (rank - 1.0) / observation_count),
        )
        for rank in ranks
    )
    bin_counts = tuple(bins.count(bin_index) for bin_index in range(FEATURE_SCORE_BIN_COUNT))
    state_count = teacher.state_count
    joint = tuple(
        tuple(
            fsum(
                probability[state]
                for bin_index, probability in zip(bins, probabilities, strict=True)
                if bin_index == current_bin
            )
            / observation_count
            for state in range(state_count)
        )
        for current_bin in range(FEATURE_SCORE_BIN_COUNT)
    )
    bin_masses = tuple(fsum(row) for row in joint)
    state_masses = tuple(fsum(row[state] for row in joint) for state in range(state_count))
    mutual_information = fsum(
        mass * log(mass / (bin_masses[bin_index] * state_masses[state]))
        for bin_index, row in enumerate(joint)
        for state, mass in enumerate(row)
        if mass > 0.0
    )
    state_entropy = -fsum(mass * log(mass) for mass in state_masses if mass > 0.0)
    ratio: float | None
    if state_entropy <= _ENTROPY_TOLERANCE:
        ratio = None
    else:
        raw_ratio = mutual_information / state_entropy
        if not isfinite(raw_ratio):
            ratio = None
        elif -FEATURE_SCORE_TIE_TOLERANCE <= raw_ratio <= 1.0 + FEATURE_SCORE_TIE_TOLERANCE:
            ratio = min(1.0, max(0.0, raw_ratio))
        else:
            ratio = None

    values_array = np.asarray(numeric_values, dtype=np.float64)
    state_weights = state_masses
    state_means: list[float] = []
    state_variances: list[float] = []
    overall_mean = float(np.mean(values_array))
    for state in range(state_count):
        state_weight = fsum(probability[state] for probability in probabilities)
        if state_weight <= 0.0:
            state_means.append(overall_mean)
            state_variances.append(0.0)
            continue
        weighted_mean = (
            fsum(
                probability[state] * value
                for value, probability in zip(numeric_values, probabilities, strict=True)
            )
            / state_weight
        )
        weighted_variance = (
            fsum(
                probability[state] * (value - weighted_mean) ** 2
                for value, probability in zip(numeric_values, probabilities, strict=True)
            )
            / state_weight
        )
        state_means.append(weighted_mean)
        state_variances.append(weighted_variance)
    between_variance = fsum(
        weight * (mean - overall_mean) ** 2
        for weight, mean in zip(state_weights, state_means, strict=True)
    )
    within_variance = fsum(
        weight * variance for weight, variance in zip(state_weights, state_variances, strict=True)
    )
    denominator = between_variance + within_variance
    eta_squared = between_variance / denominator if denominator > 0.0 else None

    reasons: list[str] = []
    if coverage < MIN_FEATURE_SCORE_COVERAGE:
        reasons.append("coverage_below_0.90")
    if observation_count < MIN_FEATURE_SCORE_OBSERVATIONS:
        reasons.append("observations_below_126")
    if state_entropy <= _ENTROPY_TOLERANCE:
        reasons.append("state_entropy_degenerate")
    if ratio is None and "state_entropy_degenerate" not in reasons:
        reasons.append("information_ratio_nonfinite")
    eligible = not reasons
    return FeatureRegimeScore(
        feature_name=feature_name,
        canonical_ordinal=canonical_ordinal,
        coverage=coverage,
        observation_count=observation_count,
        bin_counts=bin_counts,
        joint_bin_state_masses=joint,
        bin_masses=bin_masses,
        state_masses=state_masses,
        mutual_information=max(0.0, mutual_information),
        state_entropy=state_entropy,
        state_information_ratio=ratio,
        state_weights=state_weights,
        state_means=tuple(state_means),
        state_variances=tuple(state_variances),
        between_variance=max(0.0, between_variance),
        within_variance=max(0.0, within_variance),
        eta_squared=(min(1.0, max(0.0, eta_squared)) if eta_squared is not None else None),
        eligible=eligible,
        exclusion_reason=None if eligible else "; ".join(reasons),
    )


def score_raw_feature(
    snapshot: FeatureSnapshot,
    quality: QualityFilterResult,
    teacher: ProvisionalTeacherReference,
    feature_name: str,
) -> FeatureRegimeScore:
    """Score one quality-eligible raw feature against the common teacher."""

    values_by_feature, _ = _validate_inputs(snapshot, quality, teacher)
    if feature_name not in quality.eligible_features:
        raise ValueError("feature_name must belong to quality eligible_features")
    ordinal = next(
        item.canonical_ordinal for item in quality.features if item.feature_name == feature_name
    )
    return _score_feature_values(feature_name, ordinal, values_by_feature[feature_name], teacher)


def score_all_raw_features(
    snapshot: FeatureSnapshot,
    quality: QualityFilterResult,
    teacher: ProvisionalTeacherReference,
) -> tuple[FeatureRegimeScore, ...]:
    """Score every quality-eligible raw feature in canonical source order."""

    values_by_feature, _ = _validate_inputs(snapshot, quality, teacher)
    scores: list[FeatureRegimeScore] = []
    for quality_item in quality.features:
        if not quality_item.eligible:
            continue
        scores.append(
            _score_feature_values(
                quality_item.feature_name,
                quality_item.canonical_ordinal,
                values_by_feature[quality_item.feature_name],
                teacher,
            )
        )
    return tuple(scores)


compute_feature_regime_scores = score_all_raw_features
score_all_features = score_all_raw_features
score_features = score_all_raw_features


__all__ = [
    "compute_feature_regime_scores",
    "score_all_features",
    "score_all_raw_features",
    "score_features",
    "score_raw_feature",
]
