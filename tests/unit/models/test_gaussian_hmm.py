from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.special import gammaln, logsumexp  # type: ignore[import-untyped]
from scipy.stats import multivariate_t  # type: ignore[import-untyped]

from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.models.gaussian_hmm import (
    GaussianHMMSettings,
    HmmlearnGaussianHMMAdapter,
    HmmlearnGMMHMMAdapter,
    _has_material_likelihood_regression,
    _validated_em_history,
    forward_filter,
    gaussian_log_emissions,
)
from market_regime_engine.models.student_t_hmm import StudentTHMMAdapter


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


def test_student_t_log_emissions_match_scipy_reference_per_state() -> None:
    source = replace(
        artifact_2d(),
        model_family="student_t_hmm",
        degrees_of_freedom=(4.5, 12.0),
    )
    rows = np.asarray([[0.0, 0.0], [2.0, -1.0]])
    actual = gaussian_log_emissions(rows, source)
    assert source.degrees_of_freedom is not None
    for state in range(source.state_count):
        expected = multivariate_t.logpdf(
            rows,
            loc=source.means[state],
            shape=source.full_covariances[state],
            df=source.degrees_of_freedom[state],
        )
        assert actual[:, state] == pytest.approx(expected)


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
    with pytest.raises(ValueError, match="Cholesky"):
        GaussianHMMArtifact(
            state_count=2,
            feature_order=("a", "b"),
            start_probabilities=(0.5, 0.5),
            transition_matrix=((0.8, 0.2), (0.2, 0.8)),
            means=((0.0, 0.0), (1.0, 1.0)),
            full_covariances=(((1.0, 2.0), (2.0, 1.0)), ((1.0, 0.0), (0.0, 1.0))),
        )


def test_invalid_gmm_component_covariance_fails_before_emission_or_reconstruction() -> None:
    with pytest.raises(ValueError, match="mixture covariance must pass Cholesky"):
        GaussianHMMArtifact(
            state_count=2,
            feature_order=("a", "b"),
            start_probabilities=(0.5, 0.5),
            transition_matrix=((0.8, 0.2), (0.2, 0.8)),
            means=((0.0, 0.0), (1.0, 1.0)),
            full_covariances=(((1.0, 0.0), (0.0, 1.0)),) * 2,
            model_family="gmm_hmm",
            mixture_weights=((0.5, 0.5), (0.5, 0.5)),
            mixture_means=(((0.0, 0.0), (0.0, 0.0)), ((1.0, 1.0), (1.0, 1.0))),
            mixture_full_covariances=(
                (((1.0, 2.0), (2.0, 1.0)), ((1.0, 0.0), (0.0, 1.0))),
                (((1.0, 0.0), (0.0, 1.0)), ((1.0, 0.0), (0.0, 1.0))),
            ),
        )


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
    assert len(result.em_log_likelihood_history) == result.iterations
    assert result.em_log_likelihood_history
    assert np.all(np.isfinite(result.em_log_likelihood_history))


def test_em_history_validation_rejects_missing_nonfinite_and_wrong_lengths() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _validated_em_history((), 1)
    with pytest.raises(ValueError, match="length"):
        _validated_em_history((-10.0,), 2)
    with pytest.raises(ValueError, match="finite"):
        _validated_em_history((-10.0, np.nan), 2)


def test_hmmlearn_likelihood_regression_detection_allows_numerical_noise() -> None:
    assert _has_material_likelihood_regression((-100.0, -100.0 - 1e-10)) is False
    assert _has_material_likelihood_regression((-100.0, -100.1)) is True


def test_bad_rows_and_state_count_fail_closed() -> None:
    adapter = HmmlearnGaussianHMMAdapter(("a", "b"))
    with pytest.raises(ValueError, match="shape"):
        adapter.fit([[1.0]], state_count=2, seed=11)
    with pytest.raises(ValueError, match="2, 3, 4, or 5"):
        adapter.fit([[1.0, 2.0], [2.0, 3.0]], state_count=6, seed=11)


@pytest.mark.parametrize("state_count", (2, 3, 4))
def test_gmm_hmm_with_two_mixtures_extracts_and_filters_exact_mixtures(
    state_count: int,
) -> None:
    rng = np.random.default_rng(109)
    values = np.vstack(
        tuple(
            rng.normal(loc=(center, -center), scale=0.2, size=(80, 2))
            for center in np.linspace(-4.5, 4.5, num=10)
        )
    )
    adapter = HmmlearnGMMHMMAdapter(("a", "b"))
    result = adapter.fit(values, state_count=state_count, seed=11)

    assert result.artifact.model_family == "gmm_hmm"
    assert result.artifact.mixture_weights is not None
    assert all(len(weights) == 2 for weights in result.artifact.mixture_weights)
    assert np.isfinite(result.train_log_likelihood)
    assert result.em_log_likelihood_history
    assert len(result.em_log_likelihood_history) == result.iterations
    assert np.all(np.isfinite(result.em_log_likelihood_history))
    filtered = adapter.causal_filter(values[:5])
    assert np.allclose(filtered.filtered_probabilities.sum(axis=1), 1.0)
    restored = HmmlearnGMMHMMAdapter(("a", "b"))
    restored.reconstruct(result.artifact)
    assert restored.extract() == result.artifact
    with pytest.raises(ValueError, match="K=2, K=3, K=4, or K=5"):
        adapter.fit(values, state_count=6, seed=11)


def _reference_emissions(rows: np.ndarray, artifact: GaussianHMMArtifact) -> np.ndarray:
    """Independent reference densities for the three supported HMM families."""

    values = np.asarray(rows, dtype=np.float64)
    result = np.empty((len(values), artifact.state_count), dtype=np.float64)
    dimension = artifact.feature_dimension
    gaussian_constant = dimension * np.log(2.0 * np.pi)
    for state in range(artifact.state_count):
        if artifact.model_family == "gmm_hmm":
            assert artifact.mixture_weights is not None
            assert artifact.mixture_means is not None
            assert artifact.mixture_full_covariances is not None
            components: list[np.ndarray] = []
            for mixture in range(2):
                mean = np.asarray(artifact.mixture_means[state][mixture], dtype=np.float64)
                covariance = np.asarray(
                    artifact.mixture_full_covariances[state][mixture], dtype=np.float64
                )
                centered = values - mean
                solved = np.linalg.solve(covariance, centered.T).T
                quadratic = np.einsum("ij,ij->i", centered, solved)
                sign, logdet = np.linalg.slogdet(covariance)
                assert sign > 0.0
                components.append(
                    np.log(artifact.mixture_weights[state][mixture])
                    - 0.5 * (gaussian_constant + logdet + quadratic)
                )
            result[:, state] = logsumexp(np.stack(components, axis=1), axis=1)
            continue
        mean = np.asarray(artifact.means[state], dtype=np.float64)
        covariance = np.asarray(artifact.full_covariances[state], dtype=np.float64)
        centered = values - mean
        solved = np.linalg.solve(covariance, centered.T).T
        quadratic = np.einsum("ij,ij->i", centered, solved)
        sign, logdet = np.linalg.slogdet(covariance)
        assert sign > 0.0
        if artifact.model_family == "student_t_hmm":
            assert artifact.degrees_of_freedom is not None
            nu = artifact.degrees_of_freedom[state]
            result[:, state] = (
                gammaln((nu + dimension) / 2.0)
                - gammaln(nu / 2.0)
                - 0.5 * (dimension * np.log(nu * np.pi) + logdet)
                - 0.5 * (nu + dimension) * np.log1p(quadratic / nu)
            )
        else:
            result[:, state] = -0.5 * (gaussian_constant + logdet + quadratic)
    return result


def _reference_filter_score(
    rows: np.ndarray,
    artifact: GaussianHMMArtifact,
    initial_filtered_probabilities: tuple[float, ...] | None = None,
) -> tuple[float, tuple[float, ...]]:
    emissions = _reference_emissions(rows, artifact)
    transition = np.asarray(artifact.transition_matrix, dtype=np.float64)
    previous = (
        np.asarray(artifact.start_probabilities, dtype=np.float64)
        if initial_filtered_probabilities is None
        else np.asarray(initial_filtered_probabilities, dtype=np.float64)
    )
    total = 0.0
    for index, emission in enumerate(emissions):
        prior = (
            previous
            if index == 0 and initial_filtered_probabilities is None
            else previous @ transition
        )
        log_alpha = np.log(prior) + emission
        increment = float(logsumexp(log_alpha))
        total += increment
        previous = np.exp(log_alpha - increment)
    return total, tuple(float(value) for value in previous)


@pytest.mark.parametrize("family", ("gaussian", "gmm", "student_t"))
def test_train_and_oos_likelihoods_match_independent_reference(family: str) -> None:
    base = artifact_2d()
    adapter: HmmlearnGaussianHMMAdapter | HmmlearnGMMHMMAdapter | StudentTHMMAdapter
    if family == "gmm":
        artifact = replace(
            base,
            model_family="gmm_hmm",
            mixture_weights=((0.7, 0.3), (0.35, 0.65)),
            mixture_means=(
                ((-1.4, 0.6), (-0.2, 0.3)),
                ((0.7, -0.1), (1.4, -0.4)),
            ),
            mixture_full_covariances=(
                (
                    ((0.8, 0.1), (0.1, 1.4)),
                    ((1.1, 0.2), (0.2, 0.9)),
                ),
                (
                    ((0.9, -0.1), (-0.1, 1.1)),
                    ((0.6, 0.05), (0.05, 1.3)),
                ),
            ),
        )
        adapter = HmmlearnGMMHMMAdapter(base.feature_order)
    elif family == "student_t":
        artifact = replace(
            base,
            model_family="student_t_hmm",
            degrees_of_freedom=(4.5, 12.0),
        )
        adapter = StudentTHMMAdapter(base.feature_order)
    else:
        artifact = base
        adapter = HmmlearnGaussianHMMAdapter(base.feature_order)

    train = np.asarray([[-1.2, 0.4], [-0.8, 0.7], [0.6, -0.1], [1.1, -0.4]], dtype=np.float64)
    test = np.asarray([[0.9, -0.2], [-0.4, 0.5], [1.3, -0.6]], dtype=np.float64)
    adapter.reconstruct(artifact)
    expected_train, terminal = _reference_filter_score(train, artifact)
    actual_train = adapter.causal_filter(train)
    assert actual_train.log_likelihood == pytest.approx(expected_train, abs=1.0e-10)
    assert actual_train.terminal_probabilities == pytest.approx(terminal, abs=1.0e-10)
    expected_test, _ = _reference_filter_score(test, artifact, terminal)
    actual_test = adapter.causal_filter(test, actual_train.terminal_probabilities)
    assert actual_test.log_likelihood == pytest.approx(expected_test, abs=1.0e-10)
    if isinstance(adapter, (HmmlearnGaussianHMMAdapter, HmmlearnGMMHMMAdapter)):
        assert adapter.backend_reset_test_score(train) == pytest.approx(expected_train, abs=1.0e-8)
