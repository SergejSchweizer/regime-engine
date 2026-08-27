from __future__ import annotations

from dataclasses import replace

import pytest

from market_regime_engine.models.artifacts import GaussianHMMArtifact


def artifact() -> GaussianHMMArtifact:
    return GaussianHMMArtifact(
        state_count=2,
        feature_order=("a", "b"),
        start_probabilities=(0.6, 0.4),
        transition_matrix=((0.9, 0.1), (0.2, 0.8)),
        means=((-1.0, 0.5), (1.0, -0.2)),
        full_covariances=(
            ((1.0, 0.35), (0.35, 2.0)),
            ((0.7, -0.2), (-0.2, 1.2)),
        ),
    )


def test_full_covariance_off_diagonals_round_trip_unchanged() -> None:
    model = artifact()
    assert model.covariance_type == "full"
    assert model.full_covariances[0][0][1] == 0.35
    assert model.full_covariances[1][0][1] == -0.2
    assert model.feature_dimension == 2


def test_reduced_covariance_and_invalid_probability_fail_closed() -> None:
    base = artifact()
    with pytest.raises(ValueError, match="full"):
        GaussianHMMArtifact(
            state_count=base.state_count,
            feature_order=base.feature_order,
            start_probabilities=base.start_probabilities,
            transition_matrix=base.transition_matrix,
            means=base.means,
            full_covariances=base.full_covariances,
            covariance_type="diag",
        )
    with pytest.raises(ValueError, match="sum to one"):
        GaussianHMMArtifact(
            state_count=2,
            feature_order=("a",),
            start_probabilities=(0.2, 0.2),
            transition_matrix=((0.5, 0.5), (0.5, 0.5)),
            means=((0.0,), (1.0,)),
            full_covariances=(((1.0,),), ((1.0,),)),
        )


def test_asymmetric_or_near_zero_covariance_fails() -> None:
    with pytest.raises(ValueError, match="asymmetry"):
        GaussianHMMArtifact(
            state_count=2,
            feature_order=("a", "b"),
            start_probabilities=(0.5, 0.5),
            transition_matrix=((0.5, 0.5), (0.5, 0.5)),
            means=((0.0, 0.0), (1.0, 1.0)),
            full_covariances=(
                ((1.0, 0.3), (0.1, 1.0)),
                ((1.0, 0.0), (0.0, 1.0)),
            ),
        )
    with pytest.raises(ValueError, match="diagonal"):
        GaussianHMMArtifact(
            state_count=2,
            feature_order=("a",),
            start_probabilities=(0.5, 0.5),
            transition_matrix=((0.5, 0.5), (0.5, 0.5)),
            means=((0.0,), (1.0,)),
            full_covariances=(((1e-13,),), ((1.0,),)),
        )


def test_artifact_rejects_partial_or_unexpected_mixture_emissions() -> None:
    base = artifact()
    with pytest.raises(ValueError, match="cannot contain mixture"):
        replace(base, mixture_weights=((0.5, 0.5), (0.5, 0.5)))
    with pytest.raises(ValueError, match="requires complete mixture"):
        replace(base, model_family="gmm_hmm")


def test_student_t_artifact_requires_one_valid_degree_of_freedom_per_state() -> None:
    base = artifact()
    student = replace(base, model_family="student_t_hmm", degrees_of_freedom=(4.5, 9.0))
    assert student.degrees_of_freedom == (4.5, 9.0)
    with pytest.raises(ValueError, match="one degree"):
        replace(base, model_family="student_t_hmm")
    with pytest.raises(ValueError, match="greater than two"):
        replace(base, model_family="student_t_hmm", degrees_of_freedom=(2.0, 5.0))
