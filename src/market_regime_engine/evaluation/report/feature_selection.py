"""Pure, auditable explanation of the frozen feature-selection decision."""

from __future__ import annotations

from dataclasses import asdict
from math import isclose
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from market_regime_engine.evaluation.report.contracts import FeatureSelectionReport, object_payload
from market_regime_engine.feature_selection.contracts import (
    BlockSelectionEvidence,
    FeatureSelectionPolicy,
    FeatureSelectionResult,
    Stage2ConflictEvidence,
    definition_hash,
)
from market_regime_engine.feature_selection.selector import average_rank_spearman
from market_regime_engine.feature_selection.stability import FeatureSelectionStabilityDiagnostics


def _winner_score(block: BlockSelectionEvidence) -> tuple[float, float, int]:
    for score in block.scores:
        if score.feature_name == block.winner and score.eligible:
            if score.medoid_score is None:
                break
            return score.medoid_score, score.coverage, score.configured_position
    raise ValueError(f"block {block.block_id} winner lacks eligible score evidence")


def _block_report(
    first_train_rows: pd.DataFrame,
    block: BlockSelectionEvidence,
    policy: FeatureSelectionPolicy,
) -> dict[str, Any]:
    configured = next(item for item in policy.blocks if item.block_id == block.block_id)
    if tuple(score.feature_name for score in block.scores) != configured.features:
        raise ValueError(f"block {block.block_id} score order differs from policy")

    score_rows: list[dict[str, Any]] = []
    eligible_names = tuple(score.feature_name for score in block.scores if score.eligible)
    for score in block.scores:
        nonnull_count = int(first_train_rows[score.feature_name].notna().sum())
        expected_coverage = nonnull_count / len(first_train_rows)
        if not isclose(expected_coverage, score.coverage, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"coverage evidence mismatch for {score.feature_name}")
        score_rows.append(
            {
                "feature_name": score.feature_name,
                "configured_position": score.configured_position,
                "nonnull_count": nonnull_count,
                "source_row_denominator": len(first_train_rows),
                "coverage": score.coverage,
                "population_variance_ddof0": score.population_variance,
                "eligible": score.eligible,
                "exclusion_reason": score.exclusion_reason,
                "medoid_score": score.medoid_score,
            }
        )

    complete = first_train_rows.loc[:, list(eligible_names)].dropna(axis=0, how="any")
    if len(complete) != block.complete_observation_count:
        raise ValueError(f"complete-case count mismatch for block {block.block_id}")
    if len(eligible_names) == 1:
        correlation = np.ones((1, 1), dtype=np.float64)
        singleton = True
    else:
        correlation = np.abs(average_rank_spearman(complete))
        singleton = False
    distance = 1.0 - correlation

    medoid_by_name: dict[str, float] = {}
    for index, feature_name in enumerate(eligible_names):
        if len(eligible_names) == 1:
            medoid = 0.0
        else:
            medoid = float(np.mean(np.delete(distance[index], index)))
        medoid_by_name[feature_name] = medoid
        recorded = next(
            score.medoid_score for score in block.scores if score.feature_name == feature_name
        )
        if recorded is None or not isclose(
            medoid, recorded, rel_tol=0.0, abs_tol=policy.numeric_tie_abs_tolerance
        ):
            raise ValueError(f"medoid evidence mismatch for {feature_name}")

    minimum_medoid = min(medoid_by_name.values())
    medoid_tied = tuple(
        name
        for name in eligible_names
        if medoid_by_name[name] <= minimum_medoid + policy.numeric_tie_abs_tolerance
    )
    coverage_by_name = {
        score.feature_name: score.coverage for score in block.scores if score.eligible
    }
    maximum_coverage = max(coverage_by_name[name] for name in medoid_tied)
    coverage_tied = tuple(
        name
        for name in medoid_tied
        if coverage_by_name[name] >= maximum_coverage - policy.numeric_tie_abs_tolerance
    )
    position_by_name = {
        score.feature_name: score.configured_position for score in block.scores if score.eligible
    }
    resolved_winner = min(coverage_tied, key=position_by_name.__getitem__)
    if resolved_winner != block.winner:
        raise ValueError(f"Stage-1 winner trace does not reproduce block {block.block_id}")

    return {
        "block_id": block.block_id,
        "configured_features": list(configured.features),
        "feature_scores": score_rows,
        "eligible_feature_order": list(eligible_names),
        "complete_case_observation_count": block.complete_observation_count,
        "matrix_feature_order": list(eligible_names),
        "absolute_spearman_matrix": correlation.tolist(),
        "distance_matrix_one_minus_abs_spearman": distance.tolist(),
        "singleton_matrix": singleton,
        "stage1_decision": {
            "minimum_medoid_score_anchor": minimum_medoid,
            "anchored_medoid_tied_features": list(medoid_tied),
            "maximum_coverage_anchor_within_medoid_tie": maximum_coverage,
            "anchored_coverage_tied_features": list(coverage_tied),
            "configured_position_tiebreak": min(position_by_name[name] for name in coverage_tied),
            "winner": block.winner,
        },
    }


def _choose_removal(
    left: str,
    right: str,
    winner_evidence: dict[str, tuple[float, float, int]],
    tolerance: float,
) -> tuple[str, str]:
    left_score, left_coverage, left_block = winner_evidence[left]
    right_score, right_coverage, right_block = winner_evidence[right]
    score_difference = left_score - right_score
    if abs(score_difference) > tolerance:
        return (
            (left, "higher Stage-1 medoid score")
            if score_difference > 0.0
            else (right, "higher Stage-1 medoid score")
        )
    coverage_difference = left_coverage - right_coverage
    if abs(coverage_difference) > tolerance:
        return (
            (left, "lower Stage-1 coverage")
            if coverage_difference < 0.0
            else (right, "lower Stage-1 coverage")
        )
    return (
        (left, "later canonical block")
        if left_block > right_block
        else (right, "later canonical block")
    )


def _stage2_report(
    result: FeatureSelectionResult,
    policy: FeatureSelectionPolicy,
) -> dict[str, Any]:
    evidence = result.evidence
    matrix = np.asarray(evidence.stage2_abs_spearman_matrix, dtype=np.float64)
    preliminary = evidence.preliminary_medoids
    winner_evidence = {
        block.winner: (*_winner_score(block)[:2], block_index)
        for block_index, block in enumerate(evidence.block_evidence)
    }
    candidate_pairs = [
        (float(matrix[left, right]), left, right)
        for left in range(len(preliminary))
        for right in range(left + 1, len(preliminary))
        if float(matrix[left, right]) > policy.maximum_cross_block_abs_spearman
    ]
    candidate_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))

    surviving = set(preliminary)
    processed_conflicts: list[Stage2ConflictEvidence] = []
    trace: list[dict[str, Any]] = []
    removal_index: dict[str, int] = {}
    for pair_index, (value, left_index, right_index) in enumerate(candidate_pairs):
        left = preliminary[left_index]
        right = preliminary[right_index]
        before = [name for name in preliminary if name in surviving]
        if left not in surviving or right not in surviving:
            trace.append(
                {
                    "pair_index": pair_index,
                    "feature_a": left,
                    "feature_b": right,
                    "abs_spearman": value,
                    "survivors_before": before,
                    "processed": False,
                    "skip_reason": "member_already_removed",
                    "removed_feature": None,
                    "removal_reason": None,
                    "survivors_after": before,
                }
            )
            continue
        removed, reason = _choose_removal(
            left,
            right,
            winner_evidence,
            policy.numeric_tie_abs_tolerance,
        )
        surviving.remove(removed)
        conflict = Stage2ConflictEvidence(left, right, value, removed, reason)
        processed_conflicts.append(conflict)
        removal_index[removed] = pair_index
        after = [name for name in preliminary if name in surviving]
        trace.append(
            {
                "pair_index": pair_index,
                "feature_a": left,
                "feature_b": right,
                "abs_spearman": value,
                "survivors_before": before,
                "processed": True,
                "skip_reason": None,
                "removed_feature": removed,
                "removal_reason": reason,
                "survivors_after": after,
            }
        )

    final_features = tuple(name for name in preliminary if name in surviving)
    if tuple(processed_conflicts) != evidence.conflicts:
        raise ValueError("Stage-2 trace does not reproduce immutable conflict evidence")
    if final_features != evidence.final_features:
        raise ValueError("Stage-2 trace does not reproduce frozen final features")
    dispositions = [
        {
            "feature_name": name,
            "disposition": "selected" if name in surviving else "removed",
            "removal_decision_index": removal_index.get(name),
        }
        for name in preliminary
    ]
    return {
        "preliminary_medoids": list(preliminary),
        "complete_case_observation_count": evidence.stage2_complete_observation_count,
        "matrix_feature_order": list(preliminary),
        "fixed_prepruning_absolute_spearman_matrix": matrix.tolist(),
        "strict_conflict_rule": "abs_spearman > maximum_cross_block_abs_spearman",
        "maximum_cross_block_abs_spearman": policy.maximum_cross_block_abs_spearman,
        "candidate_pair_processing_order": (
            "(-abs_spearman,left_block_position,right_block_position)"
        ),
        "pair_processing_trace": trace,
        "final_dispositions": dispositions,
        "final_features": list(final_features),
    }


def build_feature_selection_report(
    first_train_rows: pd.DataFrame,
    policy: FeatureSelectionPolicy,
    result: FeatureSelectionResult,
    diagnostics: FeatureSelectionStabilityDiagnostics | None = None,
) -> FeatureSelectionReport:
    """Explain the exact frozen selection without changing or rerunning its decision."""

    if len(first_train_rows) != result.evidence.first_train_source_row_count:
        raise ValueError("first-TRAIN row count differs from frozen selection evidence")
    if policy.policy_id != result.policy_id:
        raise ValueError("feature-selection policy/result identity mismatch")
    if definition_hash(policy, result.evidence) != result.feature_selection_definition_hash:
        raise ValueError("feature-selection definition hash does not reproduce")

    blocks = [
        _block_report(first_train_rows, block, policy) for block in result.evidence.block_evidence
    ]
    payload: dict[str, Any] = {
        "first_train_source_row_count": len(first_train_rows),
        "policy": {
            "policy_id": policy.policy_id,
            "within_block_method": policy.within_block_method,
            "cross_block_method": policy.cross_block_method,
            "minimum_feature_coverage": policy.minimum_feature_coverage,
            "minimum_nonzero_variance": policy.minimum_nonzero_variance,
            "minimum_block_complete_observations": policy.minimum_block_complete_observations,
            "maximum_cross_block_abs_spearman": policy.maximum_cross_block_abs_spearman,
            "numeric_tie_abs_tolerance": policy.numeric_tie_abs_tolerance,
        },
        "stage1_blocks": blocks,
        "stage2": _stage2_report(result, policy),
        "final_features": list(result.final_features),
        "feature_selection_definition_hash": result.feature_selection_definition_hash,
        "feature_selection_execution_hash": result.feature_selection_execution_hash,
    }
    if diagnostics is not None:
        if diagnostics.frozen_final_features != result.final_features:
            raise ValueError("stability diagnostics refer to different frozen final features")
        if (
            diagnostics.frozen_definition_hash != result.feature_selection_definition_hash
            or diagnostics.frozen_execution_hash != result.feature_selection_execution_hash
        ):
            raise ValueError("stability diagnostics refer to different selection hashes")
        payload["diagnostics"] = {"diagnostic_only": True, **asdict(diagnostics)}

    return FeatureSelectionReport(
        policy_id=result.policy_id,
        final_features=result.final_features,
        feature_selection_definition_hash=result.feature_selection_definition_hash,
        feature_selection_execution_hash=result.feature_selection_execution_hash,
        evidence=object_payload(payload),
    )
