from __future__ import annotations

import pytest

from market_regime_engine.evaluations import (
    DELTA1_FEATURES,
    EvaluationId,
    EvaluationLineage,
    EvaluationResultIdentity,
    candidate_specs,
    delta1_feature_spec,
)

HASH = "a" * 64


def test_canonical_evaluation_ids_and_candidate_specs_are_exact() -> None:
    assert tuple(EvaluationId) == (
        EvaluationId.MEDOID_MULTIVARIATE,
        EvaluationId.MEDOID_UNIVARIATE,
        EvaluationId.DELTA1_UNIVARIATE,
    )
    assert len(DELTA1_FEATURES) == 13
    assert tuple(item.candidate_id for item in candidate_specs(("feature",))) == (
        "gaussian_hmm_k2_full",
        "gaussian_hmm_k3_full",
        "gaussian_hmm_k4_full",
        "gaussian_hmm_k5_full",
        "gmm_hmm_k2_m2_full",
        "gmm_hmm_k3_m2_full",
        "gmm_hmm_k4_m2_full",
        "gmm_hmm_k5_m2_full",
        "student_t_hmm_k2_full",
        "student_t_hmm_k3_full",
        "student_t_hmm_k4_full",
        "student_t_hmm_k5_full",
    )


def test_delta_spec_rejects_reordered_or_missing_features() -> None:
    with pytest.raises(ValueError, match="delta1 feature order"):
        delta1_feature_spec(tuple(reversed(DELTA1_FEATURES)))
    with pytest.raises(ValueError, match="delta1 feature order"):
        delta1_feature_spec(DELTA1_FEATURES[:-1])


def test_identity_hashes_are_separated_by_evaluation() -> None:
    multivariate = EvaluationLineage(
        EvaluationId.MEDOID_MULTIVARIATE, "build", HASH, HASH, HASH, HASH
    )
    delta = EvaluationLineage(EvaluationId.DELTA1_UNIVARIATE, "build", HASH, HASH, HASH, HASH)
    assert multivariate.definition_hash != delta.definition_hash
    identity = EvaluationResultIdentity(
        EvaluationId.DELTA1_UNIVARIATE, delta, delta1_feature_spec(DELTA1_FEATURES)
    )
    assert identity.execution_hash
