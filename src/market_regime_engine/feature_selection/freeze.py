"""Freeze first-TRAIN feature selection with fixed Stage-2 Spearman pruning."""

from __future__ import annotations

from math import isfinite

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from market_regime_engine.feature_selection.contracts import (
    BlockSelectionEvidence,
    FeatureSelectionEvidence,
    FeatureSelectionPolicy,
    FeatureSelectionResult,
    Stage2ConflictEvidence,
    definition_hash,
    execution_hash,
)
from market_regime_engine.feature_selection.selector import average_rank_spearman, select_stage1


def stage2_abs_spearman_matrix(
    first_train_rows: pd.DataFrame,
    preliminary_medoids: tuple[str, ...],
    policy: FeatureSelectionPolicy,
) -> tuple[int, tuple[tuple[float, ...], ...]]:
    """Build the one fixed complete-case 8x8 absolute-Spearman matrix."""

    if len(preliminary_medoids) != 8 or len(set(preliminary_medoids)) != 8:
        raise ValueError("Stage 2 requires exactly eight unique preliminary medoids")
    missing = tuple(
        feature for feature in preliminary_medoids if feature not in first_train_rows.columns
    )
    if missing:
        raise ValueError(f"missing Stage-2 medoid columns: {', '.join(missing)}")
    complete = first_train_rows.loc[:, list(preliminary_medoids)].dropna(axis=0, how="any")
    complete_count = len(complete)
    if complete_count < policy.minimum_block_complete_observations:
        raise ValueError(
            f"Stage 2 has {complete_count} complete observations; "
            f"requires {policy.minimum_block_complete_observations}"
        )
    values = complete.to_numpy(dtype=np.float64, copy=True)
    if not np.all(np.isfinite(values)):
        raise ValueError("Stage-2 complete-case values must be finite")
    matrix = np.abs(average_rank_spearman(complete))
    if matrix.shape != (8, 8) or not np.all(np.isfinite(matrix)):
        raise ValueError("Stage-2 absolute Spearman matrix must be finite 8x8")
    return complete_count, tuple(tuple(float(value) for value in row) for row in matrix)


def _winner_score_by_feature(
    block_evidence: tuple[BlockSelectionEvidence, ...],
) -> dict[str, tuple[float, float, int]]:
    if len(block_evidence) != 8:
        raise ValueError("Stage-2 pruning requires exactly eight block evidence records")
    result: dict[str, tuple[float, float, int]] = {}
    for block_position, block in enumerate(block_evidence):
        winner_score = next(
            (
                score
                for score in block.scores
                if score.feature_name == block.winner and score.eligible
            ),
            None,
        )
        if winner_score is None or winner_score.medoid_score is None:
            raise ValueError("every Stage-1 winner requires a finite medoid score")
        if not isfinite(winner_score.medoid_score):
            raise ValueError("Stage-1 winner medoid score must be finite")
        result[block.winner] = (
            winner_score.medoid_score,
            winner_score.coverage,
            block_position,
        )
    if len(result) != 8:
        raise ValueError("Stage-1 winners must be eight unique features")
    return result


def _removed_feature(
    feature_a: str,
    feature_b: str,
    winner_scores: dict[str, tuple[float, float, int]],
    tolerance: float,
) -> tuple[str, str]:
    score_a, coverage_a, block_a = winner_scores[feature_a]
    score_b, coverage_b, block_b = winner_scores[feature_b]
    score_difference = score_a - score_b
    if abs(score_difference) > tolerance:
        if score_difference > 0.0:
            return feature_a, "higher Stage-1 medoid score"
        return feature_b, "higher Stage-1 medoid score"
    coverage_difference = coverage_a - coverage_b
    if abs(coverage_difference) > tolerance:
        if coverage_difference < 0.0:
            return feature_a, "lower Stage-1 coverage"
        return feature_b, "lower Stage-1 coverage"
    if block_a > block_b:
        return feature_a, "later canonical block"
    return feature_b, "later canonical block"


def prune_stage2(
    block_evidence: tuple[BlockSelectionEvidence, ...],
    preliminary_medoids: tuple[str, ...],
    stage2_abs_spearman_matrix: tuple[tuple[float, ...], ...],
    policy: FeatureSelectionPolicy,
) -> tuple[tuple[Stage2ConflictEvidence, ...], tuple[str, ...]]:
    """Prune fixed-matrix conflicts without recomputation or replacement."""

    if len(preliminary_medoids) != 8 or len(set(preliminary_medoids)) != 8:
        raise ValueError("Stage-2 pruning requires exactly eight unique preliminary medoids")
    matrix = np.asarray(stage2_abs_spearman_matrix, dtype=np.float64)
    if matrix.shape != (8, 8) or not np.all(np.isfinite(matrix)):
        raise ValueError("Stage-2 pruning matrix must be finite 8x8")
    if np.any(matrix < 0.0) or np.any(matrix > 1.0):
        raise ValueError("Stage-2 absolute Spearman values must be in [0,1]")

    winner_scores = _winner_score_by_feature(block_evidence)
    if tuple(block.winner for block in block_evidence) != preliminary_medoids:
        raise ValueError("Stage-1 winners must match preliminary medoids in canonical order")

    candidates: list[tuple[float, int, int]] = []
    threshold = policy.maximum_cross_block_abs_spearman
    for left in range(8):
        for right in range(left + 1, 8):
            value = float(matrix[left, right])
            if value > threshold:
                candidates.append((value, left, right))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    surviving = set(preliminary_medoids)
    conflicts: list[Stage2ConflictEvidence] = []
    for value, left, right in candidates:
        feature_a = preliminary_medoids[left]
        feature_b = preliminary_medoids[right]
        if feature_a not in surviving or feature_b not in surviving:
            continue
        removed, reason = _removed_feature(
            feature_a,
            feature_b,
            winner_scores,
            policy.numeric_tie_abs_tolerance,
        )
        surviving.remove(removed)
        conflicts.append(
            Stage2ConflictEvidence(
                feature_a=feature_a,
                feature_b=feature_b,
                abs_spearman=value,
                removed_feature=removed,
                removal_reason=reason,
            )
        )

    final_features = tuple(feature for feature in preliminary_medoids if feature in surviving)
    if not final_features:
        raise ValueError("Stage-2 pruning cannot remove every preliminary medoid")
    return tuple(conflicts), final_features


def freeze_first_train_features(
    first_train_rows: pd.DataFrame,
    policy: FeatureSelectionPolicy,
    *,
    source_build_id: str,
    data_sha256: str,
    evaluation_plan_hash: str,
) -> FeatureSelectionResult:
    """Run and freeze canonical Stage 1 + Stage 2 using only first-TRAIN rows."""

    block_evidence = select_stage1(first_train_rows, policy)
    preliminary_medoids = tuple(block.winner for block in block_evidence)
    stage2_count, matrix = stage2_abs_spearman_matrix(
        first_train_rows,
        preliminary_medoids,
        policy,
    )
    conflicts, final_features = prune_stage2(
        block_evidence,
        preliminary_medoids,
        matrix,
        policy,
    )
    evidence = FeatureSelectionEvidence(
        first_train_source_row_count=len(first_train_rows),
        block_evidence=block_evidence,
        preliminary_medoids=preliminary_medoids,
        stage2_complete_observation_count=stage2_count,
        stage2_abs_spearman_matrix=matrix,
        conflicts=conflicts,
        final_features=final_features,
    )
    definition = definition_hash(policy, evidence)
    execution = execution_hash(
        definition,
        source_build_id=source_build_id,
        data_sha256=data_sha256,
        evaluation_plan_hash=evaluation_plan_hash,
    )
    return FeatureSelectionResult(
        policy_id=policy.policy_id,
        final_features=final_features,
        feature_selection_definition_hash=definition,
        feature_selection_execution_hash=execution,
        evidence=evidence,
    )
