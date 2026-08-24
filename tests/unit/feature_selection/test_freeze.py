from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.feature_selection import (
    BlockSelectionEvidence,
    FeatureBlock,
    FeatureScore,
    FeatureSelectionPolicy,
)
from market_regime_engine.feature_selection.freeze import (
    freeze_first_train_features,
    prune_stage2,
    stage2_abs_spearman_matrix,
)

HASH_A = "a" * 64
HASH_B = "b" * 64


def policy() -> FeatureSelectionPolicy:
    return FeatureSelectionPolicy(
        "xetra_semantic_medoid_v1",
        tuple(FeatureBlock(f"b{index}", (f"f{index}",)) for index in range(8)),
    )


def block_evidence(
    *,
    medoid_scores: tuple[float, ...] = (0.0,) * 8,
    coverages: tuple[float, ...] = (1.0,) * 8,
) -> tuple[BlockSelectionEvidence, ...]:
    return tuple(
        BlockSelectionEvidence(
            block_id=f"b{index}",
            complete_observation_count=600,
            scores=(
                FeatureScore(
                    feature_name=f"f{index}",
                    configured_position=0,
                    coverage=coverages[index],
                    population_variance=1.0,
                    medoid_score=medoid_scores[index],
                    eligible=True,
                ),
            ),
            winner=f"f{index}",
        )
        for index in range(8)
    )


def identity_matrix() -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(1.0 if row == column else 0.1 for column in range(8)) for row in range(8)
    )


def set_pair(
    matrix: tuple[tuple[float, ...], ...],
    left: int,
    right: int,
    value: float,
) -> tuple[tuple[float, ...], ...]:
    mutable = [list(row) for row in matrix]
    mutable[left][right] = value
    mutable[right][left] = value
    return tuple(tuple(row) for row in mutable)


def independent_frame(rows: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(20260824)
    return pd.DataFrame({f"f{index}": rng.normal(size=rows) for index in range(8)})


def test_exact_threshold_085_is_allowed() -> None:
    matrix = set_pair(identity_matrix(), 0, 1, 0.85)
    conflicts, final_features = prune_stage2(
        block_evidence(),
        tuple(f"f{index}" for index in range(8)),
        matrix,
        policy(),
    )
    assert conflicts == ()
    assert final_features == tuple(f"f{index}" for index in range(8))


def test_conflict_removes_higher_stage1_medoid_score() -> None:
    matrix = set_pair(identity_matrix(), 0, 1, 0.90)
    scores = (0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    conflicts, final_features = prune_stage2(
        block_evidence(medoid_scores=scores),
        tuple(f"f{index}" for index in range(8)),
        matrix,
        policy(),
    )
    assert len(conflicts) == 1
    assert conflicts[0].removed_feature == "f0"
    assert conflicts[0].removal_reason == "higher Stage-1 medoid score"
    assert "f0" not in final_features
    assert final_features[0] == "f1"


def test_ties_remove_lower_coverage_then_later_block() -> None:
    matrix = set_pair(identity_matrix(), 0, 1, 0.90)
    coverage_conflicts, _ = prune_stage2(
        block_evidence(coverages=(0.91, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
        tuple(f"f{index}" for index in range(8)),
        matrix,
        policy(),
    )
    assert coverage_conflicts[0].removed_feature == "f0"
    assert coverage_conflicts[0].removal_reason == "lower Stage-1 coverage"

    block_conflicts, _ = prune_stage2(
        block_evidence(),
        tuple(f"f{index}" for index in range(8)),
        matrix,
        policy(),
    )
    assert block_conflicts[0].removed_feature == "f1"
    assert block_conflicts[0].removal_reason == "later canonical block"


def test_conflicts_use_fixed_matrix_highest_first_and_do_not_recompute() -> None:
    matrix = identity_matrix()
    matrix = set_pair(matrix, 0, 1, 0.91)
    matrix = set_pair(matrix, 1, 2, 0.95)
    matrix = set_pair(matrix, 0, 2, 0.90)
    conflicts, final_features = prune_stage2(
        block_evidence(),
        tuple(f"f{index}" for index in range(8)),
        matrix,
        policy(),
    )
    assert conflicts[0].abs_spearman == 0.95
    assert conflicts[0].feature_a == "f1"
    assert conflicts[0].feature_b == "f2"
    assert conflicts[0].removed_feature == "f2"
    assert conflicts[1].feature_a == "f0"
    assert conflicts[1].feature_b == "f1"
    assert conflicts[1].removed_feature == "f1"
    assert "f0" in final_features
    assert "f1" not in final_features
    assert "f2" not in final_features


def test_equal_correlation_conflicts_use_canonical_pair_order() -> None:
    matrix = identity_matrix()
    matrix = set_pair(matrix, 0, 2, 0.90)
    matrix = set_pair(matrix, 0, 1, 0.90)
    conflicts, _ = prune_stage2(
        block_evidence(),
        tuple(f"f{index}" for index in range(8)),
        matrix,
        policy(),
    )
    assert (conflicts[0].feature_a, conflicts[0].feature_b) == ("f0", "f1")


def test_stage2_matrix_uses_one_fixed_complete_case_sample() -> None:
    frame = independent_frame()
    frame.loc[:19, "f0"] = np.nan
    frame.loc[20:29, "f1"] = np.nan
    count, matrix = stage2_abs_spearman_matrix(
        frame,
        tuple(f"f{index}" for index in range(8)),
        policy(),
    )
    assert count == 570
    array = np.asarray(matrix)
    assert array.shape == (8, 8)
    assert np.diag(array) == pytest.approx(np.ones(8))
    assert np.all(array >= 0.0)
    assert np.all(array <= 1.0)


def test_freeze_returns_definition_and_execution_hashes() -> None:
    result = freeze_first_train_features(
        independent_frame(),
        policy(),
        source_build_id="build-1",
        data_sha256=HASH_A,
        evaluation_plan_hash=HASH_B,
    )
    assert result.policy_id == "xetra_semantic_medoid_v1"
    assert result.final_features == tuple(f"f{index}" for index in range(8))
    assert result.evidence.preliminary_medoids == tuple(f"f{index}" for index in range(8))
    assert result.evidence.stage2_complete_observation_count == 600
    assert len(result.feature_selection_definition_hash) == 64
    assert len(result.feature_selection_execution_hash) == 64


def test_definition_is_first_train_only_while_execution_tracks_source_lineage() -> None:
    frame = independent_frame()
    first = freeze_first_train_features(
        frame,
        policy(),
        source_build_id="build-1",
        data_sha256=HASH_A,
        evaluation_plan_hash=HASH_B,
    )
    second = freeze_first_train_features(
        frame.copy(),
        policy(),
        source_build_id="build-2",
        data_sha256="c" * 64,
        evaluation_plan_hash=HASH_B,
    )
    assert first.evidence == second.evidence
    assert first.feature_selection_definition_hash == second.feature_selection_definition_hash
    assert first.feature_selection_execution_hash != second.feature_selection_execution_hash


def test_stage2_validation_fails_closed() -> None:
    medoids = tuple(f"f{index}" for index in range(8))
    with pytest.raises(ValueError, match="eight unique"):
        prune_stage2(block_evidence(), medoids[:-1], identity_matrix(), policy())
    with pytest.raises(ValueError, match="finite 8x8"):
        prune_stage2(block_evidence(), medoids, ((1.0,),), policy())
    invalid_range = set_pair(identity_matrix(), 0, 1, 1.1)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        prune_stage2(block_evidence(), medoids, invalid_range, policy())
    with pytest.raises(ValueError, match="eight block evidence"):
        prune_stage2(block_evidence()[:-1], medoids, identity_matrix(), policy())
    wrong_winner = list(block_evidence())
    wrong_winner[0] = replace(wrong_winner[0], block_id="other")
    with pytest.raises(ValueError, match="canonical order"):
        prune_stage2(tuple(reversed(wrong_winner)), medoids, identity_matrix(), policy())


def test_stage2_matrix_validation_fails_closed() -> None:
    medoids = tuple(f"f{index}" for index in range(8))
    frame = independent_frame()
    with pytest.raises(ValueError, match="eight unique"):
        stage2_abs_spearman_matrix(frame, medoids[:-1], policy())
    with pytest.raises(ValueError, match="missing Stage-2"):
        stage2_abs_spearman_matrix(frame.drop(columns=["f7"]), medoids, policy())
    too_short = independent_frame(503)
    with pytest.raises(ValueError, match="requires 504"):
        stage2_abs_spearman_matrix(too_short, medoids, policy())
