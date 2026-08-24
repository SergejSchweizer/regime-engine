"""Non-decision feature-selection stability diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from market_regime_engine.feature_selection.contracts import (
    BlockSelectionEvidence,
    FeatureSelectionPolicy,
    FeatureSelectionResult,
    Stage2ConflictEvidence,
)
from market_regime_engine.feature_selection.freeze import stage2_abs_spearman_matrix
from market_regime_engine.feature_selection.selector import select_stage1

THRESHOLD_SENSITIVITY_LEVELS = (0.80, 0.85, 0.90)
CANONICAL_STAGE2_THRESHOLD = 0.85


@dataclass(frozen=True, slots=True)
class ThresholdSensitivityDiagnostic:
    threshold: float
    selected_features: tuple[str, ...] | None
    conflicts: tuple[Stage2ConflictEvidence, ...]
    failure_reason: str | None
    canonical: bool

    def __post_init__(self) -> None:
        if self.threshold not in THRESHOLD_SENSITIVITY_LEVELS:
            raise ValueError("threshold sensitivity must use exactly 0.80, 0.85, or 0.90")
        if self.canonical != (self.threshold == CANONICAL_STAGE2_THRESHOLD):
            raise ValueError("only threshold 0.85 may be labelled canonical")
        if (self.selected_features is None) == (self.failure_reason is None):
            raise ValueError("diagnostic must contain either selected features or a failure reason")
        if self.selected_features is not None and not self.selected_features:
            raise ValueError("successful sensitivity diagnostic cannot select zero features")


@dataclass(frozen=True, slots=True)
class ShadowFoldDiagnostic:
    fold_id: str
    selected_features: tuple[str, ...] | None
    jaccard_overlap: float | None
    failure_reason: str | None

    def __post_init__(self) -> None:
        if not self.fold_id or self.fold_id.strip() != self.fold_id:
            raise ValueError("shadow fold_id must be a non-empty trimmed string")
        success = self.selected_features is not None
        if success:
            if not self.selected_features:
                raise ValueError("successful shadow diagnostic cannot select zero features")
            if self.jaccard_overlap is None or not isfinite(self.jaccard_overlap):
                raise ValueError("successful shadow diagnostic requires finite Jaccard overlap")
            if not 0.0 <= self.jaccard_overlap <= 1.0:
                raise ValueError("Jaccard overlap must be in [0,1]")
            if self.failure_reason is not None:
                raise ValueError("successful shadow diagnostic cannot carry a failure reason")
        elif self.jaccard_overlap is not None or not self.failure_reason:
            raise ValueError("failed shadow diagnostic requires only a failure reason")


@dataclass(frozen=True, slots=True)
class FeatureSelectionStabilityDiagnostics:
    frozen_final_features: tuple[str, ...]
    frozen_definition_hash: str
    frozen_execution_hash: str
    threshold_sensitivity: tuple[ThresholdSensitivityDiagnostic, ...]
    shadow_folds: tuple[ShadowFoldDiagnostic, ...]

    def __post_init__(self) -> None:
        if not self.frozen_final_features:
            raise ValueError("frozen final feature tuple cannot be empty")
        if tuple(item.threshold for item in self.threshold_sensitivity) != (
            THRESHOLD_SENSITIVITY_LEVELS
        ):
            raise ValueError("threshold sensitivity diagnostics must preserve 0.80/0.85/0.90 order")
        for value in (self.frozen_definition_hash, self.frozen_execution_hash):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError("frozen selection hashes must be lowercase SHA-256 digests")


def _winner_evidence(
    blocks: tuple[BlockSelectionEvidence, ...],
) -> dict[str, tuple[float, float, int]]:
    if len(blocks) != 8:
        raise ValueError("diagnostic Stage 2 requires exactly eight Stage-1 blocks")
    winners: dict[str, tuple[float, float, int]] = {}
    for block_position, block in enumerate(blocks):
        score = next(
            (
                candidate
                for candidate in block.scores
                if candidate.feature_name == block.winner and candidate.eligible
            ),
            None,
        )
        if score is None or score.medoid_score is None or not isfinite(score.medoid_score):
            raise ValueError("diagnostic Stage 2 requires finite Stage-1 winner evidence")
        winners[block.winner] = (score.medoid_score, score.coverage, block_position)
    if len(winners) != 8:
        raise ValueError("diagnostic Stage 2 requires eight unique Stage-1 winners")
    return winners


def _choose_removal(
    left: str,
    right: str,
    winners: dict[str, tuple[float, float, int]],
    tolerance: float,
) -> tuple[str, str]:
    left_score, left_coverage, left_block = winners[left]
    right_score, right_coverage, right_block = winners[right]
    score_difference = left_score - right_score
    if abs(score_difference) > tolerance:
        if score_difference > 0.0:
            return left, "higher Stage-1 medoid score"
        return right, "higher Stage-1 medoid score"
    coverage_difference = left_coverage - right_coverage
    if abs(coverage_difference) > tolerance:
        if coverage_difference < 0.0:
            return left, "lower Stage-1 coverage"
        return right, "lower Stage-1 coverage"
    if left_block > right_block:
        return left, "later canonical block"
    return right, "later canonical block"


def _diagnostic_prune(
    blocks: tuple[BlockSelectionEvidence, ...],
    preliminary_medoids: tuple[str, ...],
    matrix_rows: tuple[tuple[float, ...], ...],
    *,
    threshold: float,
    tolerance: float,
) -> tuple[tuple[Stage2ConflictEvidence, ...], tuple[str, ...]]:
    if threshold not in THRESHOLD_SENSITIVITY_LEVELS:
        raise ValueError("diagnostic threshold must be one of 0.80, 0.85, 0.90")
    winners_in_order = tuple(block.winner for block in blocks)
    if len(preliminary_medoids) != 8 or winners_in_order != preliminary_medoids:
        raise ValueError("diagnostic medoids must match eight Stage-1 winners in canonical order")
    matrix = np.asarray(matrix_rows, dtype=np.float64)
    if matrix.shape != (8, 8) or not np.all(np.isfinite(matrix)):
        raise ValueError("diagnostic Stage-2 matrix must be finite 8x8")
    winners = _winner_evidence(blocks)
    candidates = [
        (float(matrix[left, right]), left, right)
        for left in range(8)
        for right in range(left + 1, 8)
        if float(matrix[left, right]) > threshold
    ]
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    surviving = set(preliminary_medoids)
    conflicts: list[Stage2ConflictEvidence] = []
    for value, left_index, right_index in candidates:
        left = preliminary_medoids[left_index]
        right = preliminary_medoids[right_index]
        if left not in surviving or right not in surviving:
            continue
        removed, reason = _choose_removal(left, right, winners, tolerance)
        surviving.remove(removed)
        conflicts.append(
            Stage2ConflictEvidence(
                feature_a=left,
                feature_b=right,
                abs_spearman=value,
                removed_feature=removed,
                removal_reason=reason,
            )
        )
    selected = tuple(feature for feature in preliminary_medoids if feature in surviving)
    if not selected:
        raise ValueError("diagnostic Stage 2 cannot remove every preliminary medoid")
    return tuple(conflicts), selected


def jaccard_overlap(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    """Return exact set Jaccard overlap for two non-empty ordered feature tuples."""

    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        raise ValueError("Jaccard overlap requires at least one feature")
    return len(left_set & right_set) / len(union)


def threshold_sensitivity(
    frozen: FeatureSelectionResult,
    policy: FeatureSelectionPolicy,
) -> tuple[ThresholdSensitivityDiagnostic, ...]:
    """Rerun only Stage 2 over frozen first-TRAIN evidence at the three diagnostic thresholds."""

    evidence = frozen.evidence
    diagnostics: list[ThresholdSensitivityDiagnostic] = []
    for threshold in THRESHOLD_SENSITIVITY_LEVELS:
        try:
            conflicts, selected = _diagnostic_prune(
                evidence.block_evidence,
                evidence.preliminary_medoids,
                evidence.stage2_abs_spearman_matrix,
                threshold=threshold,
                tolerance=policy.numeric_tie_abs_tolerance,
            )
            diagnostics.append(
                ThresholdSensitivityDiagnostic(
                    threshold=threshold,
                    selected_features=selected,
                    conflicts=conflicts,
                    failure_reason=None,
                    canonical=threshold == CANONICAL_STAGE2_THRESHOLD,
                )
            )
        except Exception as exc:
            diagnostics.append(
                ThresholdSensitivityDiagnostic(
                    threshold=threshold,
                    selected_features=None,
                    conflicts=(),
                    failure_reason=f"{type(exc).__name__}: {exc}",
                    canonical=threshold == CANONICAL_STAGE2_THRESHOLD,
                )
            )
    return tuple(diagnostics)


def later_fold_shadow_diagnostics(
    later_fold_train_rows: tuple[tuple[str, pd.DataFrame], ...],
    frozen: FeatureSelectionResult,
    policy: FeatureSelectionPolicy,
) -> tuple[ShadowFoldDiagnostic, ...]:
    """Shadow-rerun selection on later TRAIN samples without affecting canonical decisions."""

    diagnostics: list[ShadowFoldDiagnostic] = []
    for fold_id, train_rows in later_fold_train_rows:
        try:
            blocks = select_stage1(train_rows, policy)
            medoids = tuple(block.winner for block in blocks)
            _, matrix = stage2_abs_spearman_matrix(train_rows, medoids, policy)
            _, selected = _diagnostic_prune(
                blocks,
                medoids,
                matrix,
                threshold=CANONICAL_STAGE2_THRESHOLD,
                tolerance=policy.numeric_tie_abs_tolerance,
            )
            diagnostics.append(
                ShadowFoldDiagnostic(
                    fold_id=fold_id,
                    selected_features=selected,
                    jaccard_overlap=jaccard_overlap(selected, frozen.final_features),
                    failure_reason=None,
                )
            )
        except Exception as exc:
            diagnostics.append(
                ShadowFoldDiagnostic(
                    fold_id=fold_id,
                    selected_features=None,
                    jaccard_overlap=None,
                    failure_reason=f"{type(exc).__name__}: {exc}",
                )
            )
    return tuple(diagnostics)


def build_stability_diagnostics(
    frozen: FeatureSelectionResult,
    policy: FeatureSelectionPolicy,
    later_fold_train_rows: tuple[tuple[str, pd.DataFrame], ...],
) -> FeatureSelectionStabilityDiagnostics:
    """Build immutable non-decision diagnostics while preserving frozen selection identity."""

    return FeatureSelectionStabilityDiagnostics(
        frozen_final_features=frozen.final_features,
        frozen_definition_hash=frozen.feature_selection_definition_hash,
        frozen_execution_hash=frozen.feature_selection_execution_hash,
        threshold_sensitivity=threshold_sensitivity(frozen, policy),
        shadow_folds=later_fold_shadow_diagnostics(later_fold_train_rows, frozen, policy),
    )
