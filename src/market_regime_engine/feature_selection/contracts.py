"""Immutable contracts for the pinned Xetra feature-selection policy."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite
from typing import Any


@dataclass(frozen=True, slots=True)
class FeatureBlock:
    block_id: str
    features: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.block_id or self.block_id.strip() != self.block_id:
            raise ValueError("block_id must be a non-empty trimmed string")
        if not self.features or len(set(self.features)) != len(self.features):
            raise ValueError("block features must be non-empty and duplicate-free")
        if any(not value or value.strip() != value for value in self.features):
            raise ValueError("feature names must be non-empty trimmed strings")


@dataclass(frozen=True, slots=True)
class FeatureSelectionPolicy:
    policy_id: str
    blocks: tuple[FeatureBlock, ...]
    within_block_method: str = "absolute_spearman_medoid"
    cross_block_method: str = "absolute_spearman_prune"
    minimum_feature_coverage: float = 0.90
    minimum_nonzero_variance: float = 1e-12
    minimum_block_complete_observations: int = 504
    maximum_cross_block_abs_spearman: float = 0.85
    numeric_tie_abs_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if self.policy_id != "xetra_semantic_medoid_v1":
            raise ValueError("unsupported feature-selection policy_id")
        if len(self.blocks) != 8 or len({block.block_id for block in self.blocks}) != 8:
            raise ValueError("policy v1 requires exactly eight unique semantic blocks")
        all_features = tuple(feature for block in self.blocks for feature in block.features)
        if len(set(all_features)) != len(all_features):
            raise ValueError("each configured feature must belong to exactly one block")
        expected = (
            self.within_block_method == "absolute_spearman_medoid",
            self.cross_block_method == "absolute_spearman_prune",
            self.minimum_feature_coverage == 0.90,
            self.minimum_nonzero_variance == 1e-12,
            self.minimum_block_complete_observations == 504,
            self.maximum_cross_block_abs_spearman == 0.85,
            self.numeric_tie_abs_tolerance == 1e-12,
        )
        if not all(expected):
            raise ValueError("feature-selection policy differs from pinned EVALUATION contract")

    @property
    def feature_universe(self) -> tuple[str, ...]:
        return tuple(feature for block in self.blocks for feature in block.features)


@dataclass(frozen=True, slots=True)
class FeatureScore:
    feature_name: str
    configured_position: int
    coverage: float
    population_variance: float
    medoid_score: float | None
    eligible: bool
    exclusion_reason: str | None = None

    def __post_init__(self) -> None:
        if self.configured_position < 0:
            raise ValueError("configured_position cannot be negative")
        if not 0.0 <= self.coverage <= 1.0 or not isfinite(self.coverage):
            raise ValueError("coverage must be finite in [0,1]")
        if not isfinite(self.population_variance):
            raise ValueError("population variance must be finite")
        if self.medoid_score is not None and not isfinite(self.medoid_score):
            raise ValueError("medoid score must be finite when supplied")
        if self.eligible == (self.exclusion_reason is not None):
            raise ValueError(
                "eligible scores have no exclusion reason; excluded scores require one"
            )


@dataclass(frozen=True, slots=True)
class BlockSelectionEvidence:
    block_id: str
    complete_observation_count: int
    scores: tuple[FeatureScore, ...]
    winner: str

    def __post_init__(self) -> None:
        eligible = {score.feature_name for score in self.scores if score.eligible}
        if self.complete_observation_count < 0:
            raise ValueError("complete observation count cannot be negative")
        if self.winner not in eligible:
            raise ValueError("block winner must be an eligible scored feature")


@dataclass(frozen=True, slots=True)
class Stage2ConflictEvidence:
    feature_a: str
    feature_b: str
    abs_spearman: float
    removed_feature: str
    removal_reason: str

    def __post_init__(self) -> None:
        if self.feature_a == self.feature_b:
            raise ValueError("Stage-2 conflict features must differ")
        if self.removed_feature not in (self.feature_a, self.feature_b):
            raise ValueError("removed feature must be one member of the conflict")
        if not isfinite(self.abs_spearman) or not 0.0 <= self.abs_spearman <= 1.0:
            raise ValueError("absolute Spearman correlation must be finite in [0,1]")
        if not self.removal_reason:
            raise ValueError("Stage-2 removal reason is required")


@dataclass(frozen=True, slots=True)
class FeatureSelectionEvidence:
    first_train_source_row_count: int
    block_evidence: tuple[BlockSelectionEvidence, ...]
    preliminary_medoids: tuple[str, ...]
    stage2_complete_observation_count: int
    stage2_abs_spearman_matrix: tuple[tuple[float, ...], ...]
    conflicts: tuple[Stage2ConflictEvidence, ...]
    final_features: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.first_train_source_row_count < 1:
            raise ValueError("first TRAIN source row count must be positive")
        if len(self.block_evidence) != 8 or len(self.preliminary_medoids) != 8:
            raise ValueError("selection evidence requires exactly eight Stage-1 blocks/medoids")
        if tuple(block.winner for block in self.block_evidence) != self.preliminary_medoids:
            raise ValueError("preliminary medoids must preserve canonical block order")
        if self.stage2_complete_observation_count < 0:
            raise ValueError("Stage-2 complete observation count cannot be negative")
        if len(self.stage2_abs_spearman_matrix) != 8 or any(
            len(row) != 8 for row in self.stage2_abs_spearman_matrix
        ):
            raise ValueError("Stage-2 evidence matrix must be fixed 8x8")
        if any(not isfinite(value) for row in self.stage2_abs_spearman_matrix for value in row):
            raise ValueError("Stage-2 matrix values must be finite")
        if not 1 <= len(self.final_features) <= 8 or len(set(self.final_features)) != len(
            self.final_features
        ):
            raise ValueError("final feature set must be duplicate-free with dimension 1..8")
        expected_order = tuple(
            feature for feature in self.preliminary_medoids if feature in set(self.final_features)
        )
        if self.final_features != expected_order:
            raise ValueError("final features must be an order-preserving preliminary-medoid subset")


@dataclass(frozen=True, slots=True)
class FeatureSelectionResult:
    policy_id: str
    final_features: tuple[str, ...]
    feature_selection_definition_hash: str
    feature_selection_execution_hash: str
    evidence: FeatureSelectionEvidence

    def __post_init__(self) -> None:
        if self.policy_id != "xetra_semantic_medoid_v1":
            raise ValueError("unsupported result policy_id")
        if self.final_features != self.evidence.final_features:
            raise ValueError("result features must match immutable selection evidence")
        for value in (
            self.feature_selection_definition_hash,
            self.feature_selection_execution_hash,
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError("selection hashes must be lowercase SHA-256 digests")


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def definition_hash(policy: FeatureSelectionPolicy, evidence: FeatureSelectionEvidence) -> str:
    payload = {"policy": asdict(policy), "evidence": asdict(evidence)}
    return sha256(canonical_json(payload)).hexdigest()


def execution_hash(
    feature_selection_definition_hash: str,
    *,
    source_build_id: str,
    data_sha256: str,
    evaluation_plan_hash: str,
) -> str:
    for name, value in (
        ("feature_selection_definition_hash", feature_selection_definition_hash),
        ("data_sha256", data_sha256),
        ("evaluation_plan_hash", evaluation_plan_hash),
    ):
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    if not source_build_id or source_build_id.strip() != source_build_id:
        raise ValueError("source_build_id must be a non-empty trimmed string")
    payload = {
        "data_sha256": data_sha256,
        "evaluation_plan_hash": evaluation_plan_hash,
        "feature_selection_definition_hash": feature_selection_definition_hash,
        "source_build_id": source_build_id,
    }
    return sha256(canonical_json(payload)).hexdigest()
