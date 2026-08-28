"""Pure identities, inputs, candidates and lineage for Xetra v3 evaluations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from hashlib import sha256
from string import hexdigits

from market_regime_engine.feature_selection.contracts import FeatureSelectionResult
from market_regime_engine.profiles.resolution import expected_candidate_ids


class EvaluationId(StrEnum):
    MEDOID_MULTIVARIATE = "medoid_multivariate"
    MEDOID_UNIVARIATE = "medoid_univariate"
    DELTA1_UNIVARIATE = "delta1_univariate"


DELTA1_FEATURES = (
    "vix_delta_1obs",
    "vix9d_delta_1obs",
    "vix3m_delta_1obs",
    "vix6m_delta_1obs",
    "vix1y_delta_1obs",
    "vstoxx_delta_1obs",
    "move_delta_1obs",
    "ciss_delta_1obs",
    "euro_hy_oas_delta_1obs",
    "us_2y_delta_1obs",
    "us_10y_delta_1obs",
    "estr_delta_1obs",
    "usd_broad_delta_1obs",
)


def _sha(value: str, name: str) -> None:
    if len(value) != 64 or value != value.lower() or any(char not in hexdigits for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _features(values: tuple[str, ...], name: str) -> None:
    if not values or len(values) != len(set(values)) or any(not value for value in values):
        raise ValueError(f"{name} must be non-empty and duplicate-free")


def _digest(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    evaluation_id: EvaluationId
    feature_order: tuple[str, ...]

    def __post_init__(self) -> None:
        _features(self.feature_order, "feature_order")
        if self.evaluation_id is EvaluationId.MEDOID_MULTIVARIATE and len(self.feature_order) > 8:
            raise ValueError("multivariate features must be a Stage-2 medoid subset")
        if self.evaluation_id is EvaluationId.MEDOID_UNIVARIATE and len(self.feature_order) != 8:
            raise ValueError("medoid univariate input requires exactly eight medoids")
        if (
            self.evaluation_id is EvaluationId.DELTA1_UNIVARIATE
            and self.feature_order != DELTA1_FEATURES
        ):
            raise ValueError("delta1 feature order must match the canonical tuple")


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    candidate_id: str
    model_family: str
    state_count: int
    mixture_count: int
    feature_order: tuple[str, ...]

    def __post_init__(self) -> None:
        _features(self.feature_order, "candidate feature_order")
        if self.state_count not in (2, 3, 4, 5):
            raise ValueError("state_count must be 2, 3, 4, or 5")
        expected = (
            f"gmm_hmm_k{self.state_count}_m2_full"
            if self.model_family == "gmm_hmm"
            else f"{self.model_family}_k{self.state_count}_full"
        )
        if self.candidate_id != expected or self.model_family not in {
            "gaussian_hmm",
            "gmm_hmm",
            "student_t_hmm",
        }:
            raise ValueError("candidate identity is unsupported")
        if self.mixture_count != (2 if self.model_family == "gmm_hmm" else 1):
            raise ValueError("candidate mixture count is inconsistent with family")


@dataclass(frozen=True, slots=True)
class EvaluationLineage:
    evaluation_id: EvaluationId
    source_build_id: str
    evaluation_plan_hash: str
    feature_selection_definition_hash: str
    feature_selection_execution_hash: str
    clock_hash: str

    def __post_init__(self) -> None:
        if not self.source_build_id or self.source_build_id.strip() != self.source_build_id:
            raise ValueError("source_build_id must be a non-empty trimmed string")
        for name in (
            "evaluation_plan_hash",
            "feature_selection_definition_hash",
            "feature_selection_execution_hash",
            "clock_hash",
        ):
            _sha(getattr(self, name), name)

    @property
    def definition_hash(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True, slots=True)
class EvaluationResultIdentity:
    evaluation_id: EvaluationId
    lineage: EvaluationLineage
    feature_spec: FeatureSpec

    def __post_init__(self) -> None:
        if (
            self.evaluation_id is not self.lineage.evaluation_id
            or self.evaluation_id is not self.feature_spec.evaluation_id
        ):
            raise ValueError("result identity evaluation IDs must agree")

    @property
    def execution_hash(self) -> str:
        return _digest(asdict(self))


def medoid_feature_spec(selection: FeatureSelectionResult) -> FeatureSpec:
    medoids = selection.evidence.preliminary_medoids
    if len(medoids) != 8 or len(set(medoids)) != 8:
        raise ValueError("selection must contain exactly eight ordered preliminary medoids")
    return FeatureSpec(EvaluationId.MEDOID_UNIVARIATE, medoids)


def multivariate_feature_spec(selection: FeatureSelectionResult) -> FeatureSpec:
    if selection.final_features != selection.evidence.final_features:
        raise ValueError("selection final features must match immutable evidence")
    return FeatureSpec(EvaluationId.MEDOID_MULTIVARIATE, selection.final_features)


def delta1_feature_spec(features: tuple[str, ...]) -> FeatureSpec:
    return FeatureSpec(EvaluationId.DELTA1_UNIVARIATE, features)


def candidate_specs(feature_order: tuple[str, ...]) -> tuple[CandidateSpec, ...]:
    specs: list[CandidateSpec] = []
    for candidate_id in expected_candidate_ids(3):
        family = "gmm_hmm" if candidate_id.startswith("gmm_") else candidate_id.rsplit("_k", 1)[0]
        state_count = int(candidate_id.split("_k", 1)[1].split("_", 1)[0])
        specs.append(
            CandidateSpec(
                candidate_id, family, state_count, 2 if family == "gmm_hmm" else 1, feature_order
            )
        )
    return tuple(specs)
