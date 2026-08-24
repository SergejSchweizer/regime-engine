from __future__ import annotations

from dataclasses import replace
from math import nan

import pytest

from market_regime_engine.feature_selection import (
    BlockSelectionEvidence,
    FeatureBlock,
    FeatureScore,
    FeatureSelectionEvidence,
    FeatureSelectionPolicy,
    FeatureSelectionResult,
    Stage2ConflictEvidence,
    canonical_json,
    definition_hash,
    execution_hash,
)

HASH_A = "a" * 64
HASH_B = "b" * 64


def policy() -> FeatureSelectionPolicy:
    return FeatureSelectionPolicy(
        policy_id="xetra_semantic_medoid_v1",
        blocks=tuple(FeatureBlock(f"block_{index}", (f"f{index}",)) for index in range(8)),
    )


def evidence() -> FeatureSelectionEvidence:
    block_evidence = tuple(
        BlockSelectionEvidence(
            block_id=f"block_{index}",
            complete_observation_count=1260,
            scores=(
                FeatureScore(
                    feature_name=f"f{index}",
                    configured_position=0,
                    coverage=1.0,
                    population_variance=1.0,
                    medoid_score=0.0,
                    eligible=True,
                ),
            ),
            winner=f"f{index}",
        )
        for index in range(8)
    )
    return FeatureSelectionEvidence(
        first_train_source_row_count=1260,
        block_evidence=block_evidence,
        preliminary_medoids=tuple(f"f{index}" for index in range(8)),
        stage2_complete_observation_count=1260,
        stage2_abs_spearman_matrix=tuple(
            tuple(1.0 if row == column else 0.1 for column in range(8)) for row in range(8)
        ),
        conflicts=(Stage2ConflictEvidence("f0", "f1", 0.9, "f1", "higher medoid score"),),
        final_features=("f0", "f2", "f3", "f4", "f5", "f6", "f7"),
    )


def test_pinned_policy_and_deterministic_hashes() -> None:
    definition = definition_hash(policy(), evidence())
    assert len(definition) == 64
    assert definition == definition_hash(policy(), evidence())
    execution = execution_hash(
        definition,
        source_build_id="build-1",
        data_sha256=HASH_A,
        evaluation_plan_hash=HASH_B,
    )
    assert len(execution) == 64
    assert execution != execution_hash(
        definition,
        source_build_id="build-2",
        data_sha256=HASH_A,
        evaluation_plan_hash=HASH_B,
    )
    result = FeatureSelectionResult(
        policy_id="xetra_semantic_medoid_v1",
        final_features=evidence().final_features,
        feature_selection_definition_hash=definition,
        feature_selection_execution_hash=execution,
        evidence=evidence(),
    )
    assert result.final_features == evidence().final_features


def test_final_subset_must_preserve_preliminary_block_order() -> None:
    with pytest.raises(ValueError, match="order-preserving"):
        replace(evidence(), final_features=("f2", "f0"))


def test_contract_validation_rejects_unpinned_policy_and_invalid_scores() -> None:
    with pytest.raises(ValueError, match="eight"):
        FeatureSelectionPolicy("xetra_semantic_medoid_v1", (FeatureBlock("one", ("f",)),))
    with pytest.raises(ValueError, match="pinned"):
        replace(policy(), maximum_cross_block_abs_spearman=0.80)
    with pytest.raises(ValueError, match="exclusion"):
        FeatureScore("f", 0, 1.0, 1.0, 0.0, False)
    with pytest.raises(ValueError, match="finite"):
        FeatureScore("f", 0, nan, 1.0, 0.0, True)


def test_hash_validation_and_canonical_json_reject_nonfinite() -> None:
    with pytest.raises(ValueError, match="data_sha256"):
        execution_hash(
            HASH_A,
            source_build_id="build",
            data_sha256="bad",
            evaluation_plan_hash=HASH_B,
        )
    with pytest.raises(ValueError):
        canonical_json({"bad": nan})
