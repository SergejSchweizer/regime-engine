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
        blocks=tuple(
            FeatureBlock(f"block_{index}", (f"f{index}",)) for index in range(8)
        ),
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
            tuple(1.0 if row == column else 0.1 for column in range(8))
            for row in range(8)
        ),
        conflicts=(
            Stage2ConflictEvidence(
                "f0",
                "f1",
                0.9,
                "f1",
                "higher medoid score",
            ),
        ),
        final_features=("f0", "f2", "f3", "f4", "f5", "f6", "f7"),
    )


def result() -> FeatureSelectionResult:
    selected = evidence()
    definition = definition_hash(policy(), selected)
    execution = execution_hash(
        definition,
        source_build_id="build-1",
        data_sha256=HASH_A,
        evaluation_plan_hash=HASH_B,
    )
    return FeatureSelectionResult(
        policy_id="xetra_semantic_medoid_v1",
        final_features=selected.final_features,
        feature_selection_definition_hash=definition,
        feature_selection_execution_hash=execution,
        evidence=selected,
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
    assert result().final_features == evidence().final_features
    assert policy().feature_universe == tuple(f"f{index}" for index in range(8))


def test_feature_block_validation_paths() -> None:
    for block_id, features, match in (
        ("", ("f",), "block_id"),
        (" bad", ("f",), "block_id"),
        ("block", (), "duplicate-free"),
        ("block", ("f", "f"), "duplicate-free"),
        ("block", (" bad",), "feature names"),
    ):
        with pytest.raises(ValueError, match=match):
            FeatureBlock(block_id, features)


def test_policy_validation_paths() -> None:
    with pytest.raises(ValueError, match="policy_id"):
        replace(policy(), policy_id="other")
    with pytest.raises(ValueError, match="eight"):
        FeatureSelectionPolicy(
            "xetra_semantic_medoid_v1",
            (FeatureBlock("one", ("f",)),),
        )
    duplicate_blocks = tuple(
        FeatureBlock("same", (f"f{index}",)) for index in range(8)
    )
    with pytest.raises(ValueError, match="eight"):
        FeatureSelectionPolicy("xetra_semantic_medoid_v1", duplicate_blocks)
    duplicate_feature_blocks = tuple(
        FeatureBlock(
            f"block_{index}",
            ("shared",) if index < 2 else (f"f{index}",),
        )
        for index in range(8)
    )
    with pytest.raises(ValueError, match="exactly one block"):
        FeatureSelectionPolicy("xetra_semantic_medoid_v1", duplicate_feature_blocks)
    for field_name, value in (
        ("within_block_method", "other"),
        ("cross_block_method", "other"),
        ("minimum_feature_coverage", 0.91),
        ("minimum_nonzero_variance", 2e-12),
        ("minimum_block_complete_observations", 505),
        ("maximum_cross_block_abs_spearman", 0.80),
        ("numeric_tie_abs_tolerance", 2e-12),
    ):
        with pytest.raises(ValueError, match="pinned"):
            replace(policy(), **{field_name: value})


def test_feature_score_validation_paths() -> None:
    valid = FeatureScore("f", 0, 1.0, 1.0, 0.0, True)
    assert valid.exclusion_reason is None
    cases = (
        ({"configured_position": -1}, "configured_position"),
        ({"coverage": -0.1}, "coverage"),
        ({"coverage": 1.1}, "coverage"),
        ({"coverage": nan}, "coverage"),
        ({"population_variance": nan}, "variance"),
        ({"medoid_score": nan}, "medoid"),
        ({"eligible": False}, "exclusion"),
        ({"exclusion_reason": "bad"}, "exclusion"),
    )
    for changes, match in cases:
        with pytest.raises(ValueError, match=match):
            replace(valid, **changes)
    excluded = FeatureScore(
        "f",
        0,
        0.5,
        0.0,
        None,
        False,
        "coverage",
    )
    assert excluded.exclusion_reason == "coverage"


def test_block_and_conflict_evidence_validation_paths() -> None:
    valid_score = FeatureScore("f", 0, 1.0, 1.0, 0.0, True)
    valid_block = BlockSelectionEvidence("b", 504, (valid_score,), "f")
    assert valid_block.winner == "f"
    with pytest.raises(ValueError, match="negative"):
        replace(valid_block, complete_observation_count=-1)
    with pytest.raises(ValueError, match="winner"):
        replace(valid_block, winner="other")

    conflict = Stage2ConflictEvidence("a", "b", 0.9, "b", "later block")
    assert conflict.removed_feature == "b"
    with pytest.raises(ValueError, match="differ"):
        replace(conflict, feature_b="a")
    with pytest.raises(ValueError, match="one member"):
        replace(conflict, removed_feature="c")
    for value in (-0.1, 1.1, nan):
        with pytest.raises(ValueError, match="Spearman"):
            replace(conflict, abs_spearman=value)
    with pytest.raises(ValueError, match="reason"):
        replace(conflict, removal_reason="")


def test_selection_evidence_validation_paths() -> None:
    valid = evidence()
    with pytest.raises(ValueError, match="source row"):
        replace(valid, first_train_source_row_count=0)
    with pytest.raises(ValueError, match="eight"):
        replace(valid, block_evidence=valid.block_evidence[:-1])
    with pytest.raises(ValueError, match="canonical block order"):
        replace(
            valid,
            preliminary_medoids=tuple(reversed(valid.preliminary_medoids)),
        )
    with pytest.raises(ValueError, match="cannot be negative"):
        replace(valid, stage2_complete_observation_count=-1)
    with pytest.raises(ValueError, match="8x8"):
        replace(valid, stage2_abs_spearman_matrix=((1.0,),))
    matrix = [list(row) for row in valid.stage2_abs_spearman_matrix]
    matrix[0][0] = nan
    with pytest.raises(ValueError, match="finite"):
        replace(
            valid,
            stage2_abs_spearman_matrix=tuple(tuple(row) for row in matrix),
        )
    for features in ((), ("f0", "f0")):
        with pytest.raises(ValueError, match="duplicate-free"):
            replace(valid, final_features=features)
    with pytest.raises(ValueError, match="order-preserving"):
        replace(valid, final_features=("f2", "f0"))


def test_result_validation_paths() -> None:
    valid = result()
    with pytest.raises(ValueError, match="policy_id"):
        replace(valid, policy_id="other")
    with pytest.raises(ValueError, match="match immutable"):
        replace(valid, final_features=("f0",))
    with pytest.raises(ValueError, match="SHA-256"):
        replace(valid, feature_selection_definition_hash="bad")
    with pytest.raises(ValueError, match="SHA-256"):
        replace(valid, feature_selection_execution_hash="G" * 64)


def test_hash_validation_and_canonical_json_reject_nonfinite() -> None:
    definition = definition_hash(policy(), evidence())
    for field_name, values in (
        ("feature_selection_definition_hash", ("bad", HASH_A, HASH_B)),
        ("data_sha256", (definition, "bad", HASH_B)),
        ("evaluation_plan_hash", (definition, HASH_A, "bad")),
    ):
        with pytest.raises(ValueError, match=field_name):
            execution_hash(
                values[0],
                source_build_id="build",
                data_sha256=values[1],
                evaluation_plan_hash=values[2],
            )
    with pytest.raises(ValueError, match="source_build_id"):
        execution_hash(
            definition,
            source_build_id="",
            data_sha256=HASH_A,
            evaluation_plan_hash=HASH_B,
        )
    with pytest.raises(ValueError):
        canonical_json({"bad": nan})
