from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.models.gaussian_hmm import (
    GaussianHMMSettings,
    HmmlearnGaussianHMMAdapter,
    forward_filter,
    gaussian_log_emissions,
)


def artifact_2d() -> GaussianHMMArtifact:
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


def test_settings_are_exact_and_reduced_covariance_is_rejected() -> None:
    settings = GaussianHMMSettings()
    assert settings.backend == "hmmlearn==0.3.3"
    assert settings.n_iter == 1000
    assert settings.tol == 1e-4
    assert settings.min_covar == 1e-6
    assert settings.params == settings.init_params == "stmc"
    with pytest.raises(ValueError, match="pinned"):
        replace(settings, covariance_type="diag")


def test_off_diagonal_artifact_round_trip_and_forward_primitives() -> None:
    source = artifact_2d()
    adapter = HmmlearnGaussianHMMAdapter(source.feature_order)
    adapter.reconstruct(source)
    restored = adapter.extract()
    assert restored == source
    assert restored.full_covariances[0][0][1] == 0.35
    emissions = gaussian_log_emissions([[0.0, 0.0], [0.1, -0.1]], restored)
    assert emissions.shape == (2, 2)
    filtered = forward_filter([[0.0, 0.0], [0.1, -0.1]], restored)
    assert filtered.filtered_probabilities.shape == (2, 2)
    assert np.allclose(filtered.filtered_probabilities.sum(axis=1), 1.0)
    assert np.isfinite(filtered.log_likelihood)


def test_train_continuation_is_distinct_from_backend_reset_test_score() -> None:
    source = GaussianHMMArtifact(
        state_count=2,
        feature_order=("x",),
        start_probabilities=(0.99, 0.01),
        transition_matrix=((0.99, 0.01), (0.01, 0.99)),
        means=((-5.0,), (5.0,)),
        full_covariances=(((1.0,),), ((1.0,),)),
    )
    adapter = HmmlearnGaussianHMMAdapter(("x",))
    adapter.reconstruct(source)
    continuation = adapter.score_continuation([[5.0]], (0.0, 1.0))
    reset = adapter.backend_reset_test_score([[5.0]])
    assert continuation > reset
    assert continuation == pytest.approx(adapter.causal_filter([[5.0]], (0.0, 1.0)).log_likelihood)


def test_non_positive_definite_covariance_fails_without_jitter() -> None:
    invalid = GaussianHMMArtifact(
        state_count=2,
        feature_order=("a", "b"),
        start_probabilities=(0.5, 0.5),
        transition_matrix=((0.8, 0.2), (0.2, 0.8)),
        means=((0.0, 0.0), (1.0, 1.0)),
        full_covariances=(
            ((1.0, 2.0), (2.0, 1.0)),
            ((1.0, 0.0), (0.0, 1.0)),
        ),
    )
    with pytest.raises(ValueError, match="Cholesky"):
        HmmlearnGaussianHMMAdapter(("a", "b")).reconstruct(invalid)


def test_k2_fit_smoke_extracts_full_covariance() -> None:
    rng = np.random.default_rng(7)
    left = rng.normal(loc=-2.0, scale=0.4, size=(120, 2))
    right = rng.normal(loc=2.0, scale=0.5, size=(120, 2))
    values = np.vstack((left, right))
    adapter = HmmlearnGaussianHMMAdapter(("a", "b"))
    result = adapter.fit(values, state_count=2, seed=11)
    assert result.seed == 11
    assert np.isfinite(result.train_log_likelihood)
    assert result.artifact.covariance_type == "full"
    assert result.artifact.state_count == 2
    assert result.artifact.feature_dimension == 2


def test_bad_rows_and_state_count_fail_closed() -> None:
    adapter = HmmlearnGaussianHMMAdapter(("a", "b"))
    with pytest.raises(ValueError, match="shape"):
        adapter.fit([[1.0]], state_count=2, seed=11)
    with pytest.raises(ValueError, match="2, 3, or 4"):
        adapter.fit([[1.0, 2.0], [2.0, 3.0]], state_count=5, seed=11)
