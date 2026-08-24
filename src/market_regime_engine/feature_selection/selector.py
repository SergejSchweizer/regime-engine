"""Pure first-stage absolute-Spearman medoid selection."""

from __future__ import annotations

from math import isfinite

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from market_regime_engine.feature_selection.contracts import (
    BlockSelectionEvidence,
    FeatureBlock,
    FeatureScore,
    FeatureSelectionPolicy,
)


def average_rank_spearman(values: pd.DataFrame) -> np.ndarray:
    """Compute Spearman correlation as Pearson correlation of average-rank columns."""

    if values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("Spearman matrix requires at least two rows and two columns")
    numeric = values.to_numpy(dtype=np.float64, copy=True)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("Spearman complete-case values must be finite")
    ranked = values.rank(axis=0, method="average").to_numpy(dtype=np.float64)
    matrix = np.asarray(np.corrcoef(ranked, rowvar=False), dtype=np.float64)
    if matrix.shape != (values.shape[1], values.shape[1]) or not np.all(np.isfinite(matrix)):
        raise ValueError("required Spearman correlations must be finite")
    return matrix


def _eligible_score(
    frame: pd.DataFrame,
    feature_name: str,
    configured_position: int,
    policy: FeatureSelectionPolicy,
) -> FeatureScore:
    series = frame[feature_name]
    source_rows = len(frame)
    non_null = series.dropna()
    coverage = len(non_null) / source_rows
    raw = non_null.to_numpy(dtype=np.float64, copy=True)

    if coverage < policy.minimum_feature_coverage:
        variance = float(np.var(raw, ddof=0)) if raw.size and np.all(np.isfinite(raw)) else 0.0
        return FeatureScore(
            feature_name,
            configured_position,
            coverage,
            variance,
            None,
            False,
            "coverage_below_minimum",
        )
    if not np.all(np.isfinite(raw)):
        return FeatureScore(
            feature_name,
            configured_position,
            coverage,
            0.0,
            None,
            False,
            "nonfinite_nonnull_value",
        )
    variance = float(np.var(raw, ddof=0)) if raw.size else 0.0
    if not isfinite(variance):
        raise ValueError("population variance must be finite")
    if variance <= policy.minimum_nonzero_variance:
        return FeatureScore(
            feature_name,
            configured_position,
            coverage,
            variance,
            None,
            False,
            "variance_below_or_equal_minimum",
        )
    return FeatureScore(
        feature_name,
        configured_position,
        coverage,
        variance,
        0.0,
        True,
    )


def _candidate_is_better(
    candidate: FeatureScore,
    current: FeatureScore,
    tolerance: float,
) -> bool:
    if candidate.medoid_score is None or current.medoid_score is None:
        raise ValueError("eligible candidate comparison requires medoid scores")
    score_difference = candidate.medoid_score - current.medoid_score
    if abs(score_difference) > tolerance:
        return score_difference < 0.0
    coverage_difference = candidate.coverage - current.coverage
    if abs(coverage_difference) > tolerance:
        return coverage_difference > 0.0
    return candidate.configured_position < current.configured_position


def select_stage1_block(
    first_train_rows: pd.DataFrame,
    block: FeatureBlock,
    policy: FeatureSelectionPolicy,
) -> BlockSelectionEvidence:
    """Select exactly one Stage-1 medoid for one configured semantic block."""

    if len(first_train_rows) < 1:
        raise ValueError("first TRAIN rows must be non-empty")
    missing = tuple(
        feature for feature in block.features if feature not in first_train_rows.columns
    )
    if missing:
        raise ValueError(f"missing configured feature columns: {', '.join(missing)}")

    initial_scores = tuple(
        _eligible_score(first_train_rows, feature, position, policy)
        for position, feature in enumerate(block.features)
    )
    eligible = tuple(score for score in initial_scores if score.eligible)
    if not eligible:
        raise ValueError(f"semantic block {block.block_id} has no eligible features")

    eligible_names = [score.feature_name for score in eligible]
    complete = first_train_rows.loc[:, eligible_names].dropna(axis=0, how="any")
    complete_count = len(complete)
    if complete_count < policy.minimum_block_complete_observations:
        raise ValueError(
            f"semantic block {block.block_id} has {complete_count} complete observations; "
            f"requires {policy.minimum_block_complete_observations}"
        )
    complete_values = complete.to_numpy(dtype=np.float64, copy=True)
    if not np.all(np.isfinite(complete_values)):
        raise ValueError("eligible block-complete values must be finite")

    medoid_by_name: dict[str, float] = {}
    if len(eligible) == 1:
        medoid_by_name[eligible[0].feature_name] = 0.0
    else:
        correlations = average_rank_spearman(complete)
        distances = 1.0 - np.abs(correlations)
        for index, score in enumerate(eligible):
            other_distances = np.delete(distances[index], index)
            medoid = float(np.mean(other_distances))
            if not isfinite(medoid):
                raise ValueError("medoid score must be finite")
            medoid_by_name[score.feature_name] = medoid

    final_scores = tuple(
        FeatureScore(
            feature_name=score.feature_name,
            configured_position=score.configured_position,
            coverage=score.coverage,
            population_variance=score.population_variance,
            medoid_score=medoid_by_name[score.feature_name] if score.eligible else None,
            eligible=score.eligible,
            exclusion_reason=score.exclusion_reason,
        )
        for score in initial_scores
    )
    eligible_final = tuple(score for score in final_scores if score.eligible)
    winner = eligible_final[0]
    for candidate in eligible_final[1:]:
        if _candidate_is_better(candidate, winner, policy.numeric_tie_abs_tolerance):
            winner = candidate
    return BlockSelectionEvidence(
        block_id=block.block_id,
        complete_observation_count=complete_count,
        scores=final_scores,
        winner=winner.feature_name,
    )


def select_stage1(
    first_train_rows: pd.DataFrame,
    policy: FeatureSelectionPolicy,
) -> tuple[BlockSelectionEvidence, ...]:
    """Run Stage 1 for all eight blocks in canonical policy order."""

    return tuple(select_stage1_block(first_train_rows, block, policy) for block in policy.blocks)
