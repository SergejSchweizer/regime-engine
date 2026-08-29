"""Pinned hmmlearn full-covariance Gaussian HMM adapter and causal forward primitives."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from math import isfinite

import numpy as np
import numpy.typing as npt
from hmmlearn.hmm import GMMHMM, GaussianHMM  # type: ignore[import-untyped]
from scipy.special import gammaln  # type: ignore[import-untyped]

from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.models.protocols import ArrayF64, FilterResult, FitResult

_PROB_TOL = 1e-10
_HMMLEARN_REGRESSION_TOLERANCE = float(np.finfo(np.float64).eps ** 0.5)


def _has_material_likelihood_regression(history: Iterable[float]) -> bool:
    values = tuple(float(value) for value in history)
    return any(
        current - previous < -_HMMLEARN_REGRESSION_TOLERANCE
        for previous, current in pairwise(values)
    )


@dataclass(frozen=True, slots=True)
class GaussianHMMSettings:
    backend: str = "hmmlearn==0.3.3"
    covariance_type: str = "full"
    implementation: str = "log"
    n_iter: int = 1000
    tol: float = 1e-4
    min_covar: float = 1e-6
    startprob_prior: float = 1.0
    transmat_prior: float = 1.0
    means_prior: float = 0.0
    means_weight: float = 0.0
    covars_prior: float = 0.01
    covars_weight: float = 1.0
    params: str = "stmc"
    init_params: str = "stmc"

    def __post_init__(self) -> None:
        expected = (
            self.backend == "hmmlearn==0.3.3",
            self.covariance_type == "full",
            self.implementation == "log",
            self.n_iter == 1000,
            self.tol == 1e-4,
            self.min_covar == 1e-6,
            self.startprob_prior == 1.0,
            self.transmat_prior == 1.0,
            self.means_prior == 0.0,
            self.means_weight == 0.0,
            self.covars_prior == 0.01,
            self.covars_weight == 1.0,
            self.params == "stmc",
            self.init_params == "stmc",
        )
        if not all(expected):
            raise ValueError("Gaussian HMM settings differ from pinned EVALUATION contract")


def _rows(values: npt.ArrayLike, dimension: int) -> ArrayF64:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] < 1 or result.shape[1] != dimension:
        raise ValueError("HMM rows must have shape (n, exact feature dimension)")
    if not np.all(np.isfinite(result)):
        raise ValueError("HMM rows must be finite")
    return result


def _probability_vector(values: npt.ArrayLike, state_count: int, name: str) -> ArrayF64:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (state_count,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} has invalid shape or nonfinite values")
    if np.any(result < 0.0) or not np.isclose(float(result.sum()), 1.0, rtol=0.0, atol=_PROB_TOL):
        raise ValueError(f"{name} must be nonnegative and normalized within 1e-10")
    return result


def _validate_positive_definite(artifact: GaussianHMMArtifact) -> None:
    for covariance in artifact.full_covariances:
        matrix = np.asarray(covariance, dtype=np.float64)
        symmetric = (matrix + matrix.T) / 2.0
        try:
            np.linalg.cholesky(symmetric)
        except np.linalg.LinAlgError as exc:
            raise ValueError("full covariance must pass Cholesky without jitter") from exc


def _validate_mixture_covariances(artifact: GaussianHMMArtifact) -> None:
    if artifact.model_family != "gmm_hmm":
        return
    assert artifact.mixture_full_covariances is not None
    for state_covariances in artifact.mixture_full_covariances:
        for covariance in state_covariances:
            matrix = np.asarray(covariance, dtype=np.float64)
            if matrix.shape != (artifact.feature_dimension, artifact.feature_dimension):
                raise ValueError("mixture covariance must have exact d x d shape")
            if not np.all(np.isfinite(matrix)):
                raise ValueError("mixture covariance values must be finite")
            if float(np.max(np.abs(matrix - matrix.T))) > _PROB_TOL:
                raise ValueError("mixture covariance asymmetry exceeds 1e-10")
            if np.any(np.diag(matrix) < 1e-12):
                raise ValueError("mixture covariance diagonal variance is below 1e-12")
            try:
                np.linalg.cholesky((matrix + matrix.T) / 2.0)
            except np.linalg.LinAlgError as exc:
                raise ValueError("mixture covariance must pass Cholesky without jitter") from exc


def gaussian_log_emissions(rows: npt.ArrayLike, artifact: GaussianHMMArtifact) -> ArrayF64:
    """Return exact Gaussian, GMM, or multivariate Student-t log densities by state."""
    values = _rows(rows, artifact.feature_dimension)
    _validate_positive_definite(artifact)
    _validate_mixture_covariances(artifact)
    output = np.empty((values.shape[0], artifact.state_count), dtype=np.float64)
    constant = artifact.feature_dimension * np.log(2.0 * np.pi)
    for state in range(artifact.state_count):
        if artifact.model_family == "gmm_hmm":
            assert artifact.mixture_weights is not None
            assert artifact.mixture_means is not None
            assert artifact.mixture_full_covariances is not None
            mixture_log_density = np.empty((values.shape[0], 2), dtype=np.float64)
            for mixture in range(2):
                mean = np.asarray(artifact.mixture_means[state][mixture], dtype=np.float64)
                covariance = np.asarray(
                    artifact.mixture_full_covariances[state][mixture], dtype=np.float64
                )
                covariance = (covariance + covariance.T) / 2.0
                sign, logdet = np.linalg.slogdet(covariance)
                if sign <= 0.0 or not isfinite(float(logdet)):
                    raise ValueError("mixture covariance determinant must be finite and positive")
                centered = values - mean
                solved = np.linalg.solve(covariance, centered.T).T
                quadratic = np.einsum("ij,ij->i", centered, solved)
                mixture_log_density[:, mixture] = np.log(
                    artifact.mixture_weights[state][mixture]
                ) - 0.5 * (constant + logdet + quadratic)
            maximum = np.max(mixture_log_density, axis=1, keepdims=True)
            output[:, state] = maximum[:, 0] + np.log(
                np.exp(mixture_log_density - maximum).sum(axis=1)
            )
            continue
        mean = np.asarray(artifact.means[state], dtype=np.float64)
        covariance = np.asarray(artifact.full_covariances[state], dtype=np.float64)
        covariance = (covariance + covariance.T) / 2.0
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0.0 or not isfinite(float(logdet)):
            raise ValueError("full covariance determinant must be finite and positive")
        centered = values - mean
        solved = np.linalg.solve(covariance, centered.T).T
        quadratic = np.einsum("ij,ij->i", centered, solved)
        if artifact.model_family == "student_t_hmm":
            assert artifact.degrees_of_freedom is not None
            nu = artifact.degrees_of_freedom[state]
            dimension = artifact.feature_dimension
            output[:, state] = (
                gammaln((nu + dimension) / 2.0)
                - gammaln(nu / 2.0)
                - 0.5 * (dimension * np.log(nu * np.pi) + logdet)
                - 0.5 * (nu + dimension) * np.log1p(quadratic / nu)
            )
        else:
            output[:, state] = -0.5 * (constant + logdet + quadratic)
    if not np.all(np.isfinite(output)):
        raise ValueError("Gaussian emission log densities must be finite")
    return output


def _normalize_log_weights(log_weights: ArrayF64) -> tuple[ArrayF64, float]:
    maximum = float(np.max(log_weights))
    shifted = np.exp(log_weights - maximum)
    normalizer = float(shifted.sum())
    if not isfinite(maximum) or not isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("forward normalization failed")
    probabilities = shifted / normalizer
    return probabilities, maximum + float(np.log(normalizer))


def forward_filter(
    rows: npt.ArrayLike,
    artifact: GaussianHMMArtifact,
    *,
    terminal_train_alpha: tuple[float, ...] | None = None,
) -> FilterResult:
    """Causal stabilized alpha recursion; supplied alpha means TRAIN continuation."""
    emissions = gaussian_log_emissions(rows, artifact)
    transition = np.asarray(artifact.transition_matrix, dtype=np.float64)
    filtered = np.empty_like(emissions)
    log_likelihood = 0.0

    previous: ArrayF64 | None = None
    if terminal_train_alpha is not None:
        previous = _probability_vector(
            terminal_train_alpha, artifact.state_count, "terminal_train_alpha"
        )

    for index in range(emissions.shape[0]):
        if previous is None:
            prior = _probability_vector(
                artifact.start_probabilities, artifact.state_count, "start_probabilities"
            )
        else:
            prior = previous @ transition
            prior = _probability_vector(prior, artifact.state_count, "forward prior")
        with np.errstate(divide="ignore"):
            log_prior = np.log(prior)
        alpha, increment = _normalize_log_weights(log_prior + emissions[index])
        filtered[index] = alpha
        log_likelihood += increment
        previous = alpha

    terminal = tuple(float(value) for value in filtered[-1])
    return FilterResult(
        filtered_probabilities=filtered,
        log_likelihood=float(log_likelihood),
        terminal_probabilities=terminal,
    )


class HmmlearnGaussianHMMAdapter:
    """Backend adapter restricted to the exact Gaussian MVP contract."""

    def __init__(
        self,
        feature_order: tuple[str, ...],
        settings: GaussianHMMSettings | None = None,
    ) -> None:
        if not feature_order or len(set(feature_order)) != len(feature_order):
            raise ValueError("feature_order must be non-empty and duplicate-free")
        self.feature_order = feature_order
        self.settings = settings or GaussianHMMSettings()
        self._model: GaussianHMM | GMMHMM | None = None
        self._artifact: GaussianHMMArtifact | None = None

    def _new_model(self, state_count: int, seed: int) -> GaussianHMM | GMMHMM:
        if state_count not in (2, 3, 4, 5):
            raise ValueError("Gaussian MVP state_count must be 2, 3, 4, or 5")
        settings = self.settings
        return GaussianHMM(
            n_components=state_count,
            covariance_type=settings.covariance_type,
            min_covar=settings.min_covar,
            startprob_prior=settings.startprob_prior,
            transmat_prior=settings.transmat_prior,
            means_prior=settings.means_prior,
            means_weight=settings.means_weight,
            covars_prior=settings.covars_prior,
            covars_weight=settings.covars_weight,
            algorithm="viterbi",
            random_state=seed,
            n_iter=settings.n_iter,
            tol=settings.tol,
            params=settings.params,
            verbose=False,
            implementation=settings.implementation,
            init_params=settings.init_params,
        )

    def fit(self, train_rows: npt.ArrayLike, state_count: int, seed: int) -> FitResult:
        values = _rows(train_rows, len(self.feature_order))
        model = self._new_model(state_count, seed)
        # Zero-probability mixture components are valid unused components.
        # hmmlearn logs their mathematical log(0), while our artifacts preserve
        # the exact zero-probability semantics without treating them as failures.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="divide by zero encountered in log",
                category=RuntimeWarning,
                module=r"hmmlearn\\._emissions",
            )
            hmmlearn_logger = logging.getLogger("hmmlearn.base")
            was_disabled = hmmlearn_logger.disabled
            hmmlearn_logger.disabled = True
            try:
                model.fit(values)
            finally:
                hmmlearn_logger.disabled = was_disabled
        self._model = model
        artifact = self._extract_model(model)
        _validate_positive_definite(artifact)
        self._artifact = artifact
        train_log_likelihood = float(model.score(values))
        if not isfinite(train_log_likelihood):
            raise ValueError("TRAIN log likelihood must be finite")
        return FitResult(
            artifact=artifact,
            train_log_likelihood=train_log_likelihood,
            converged=bool(model.monitor_.converged)
            and not _has_material_likelihood_regression(model.monitor_.history),
            iterations=int(model.monitor_.iter),
            seed=seed,
        )

    def _extract_model(self, model: GaussianHMM | GMMHMM) -> GaussianHMMArtifact:
        if not isinstance(model, GaussianHMM):
            raise TypeError("Gaussian adapter requires a GaussianHMM backend model")
        artifact = GaussianHMMArtifact(
            state_count=int(model.n_components),
            feature_order=self.feature_order,
            start_probabilities=tuple(float(value) for value in model.startprob_),
            transition_matrix=tuple(
                tuple(float(value) for value in row) for row in model.transmat_
            ),
            means=tuple(tuple(float(value) for value in row) for row in model.means_),
            full_covariances=tuple(
                tuple(tuple(float(value) for value in row) for row in matrix)
                for matrix in model.covars_
            ),
        )
        _validate_positive_definite(artifact)
        return artifact

    def extract(self) -> GaussianHMMArtifact:
        if self._artifact is None:
            raise RuntimeError("no fitted or reconstructed Gaussian HMM is available")
        return self._artifact

    def reconstruct(self, artifact: GaussianHMMArtifact) -> None:
        if artifact.feature_order != self.feature_order:
            raise ValueError("artifact feature order does not match adapter feature order")
        _validate_positive_definite(artifact)
        model = self._new_model(artifact.state_count, seed=0)
        model.startprob_ = np.asarray(artifact.start_probabilities, dtype=np.float64)
        model.transmat_ = np.asarray(artifact.transition_matrix, dtype=np.float64)
        model.means_ = np.asarray(artifact.means, dtype=np.float64)
        model.covars_ = np.asarray(artifact.full_covariances, dtype=np.float64)
        model.n_features = artifact.feature_dimension
        self._model = model
        self._artifact = artifact

    def causal_filter(
        self,
        rows: npt.ArrayLike,
        initial_filtered_probabilities: tuple[float, ...] | None = None,
    ) -> FilterResult:
        return forward_filter(
            rows,
            self.extract(),
            terminal_train_alpha=initial_filtered_probabilities,
        )

    def score_continuation(
        self,
        test_rows: npt.ArrayLike,
        terminal_train_alpha: tuple[float, ...],
    ) -> float:
        return self.causal_filter(test_rows, terminal_train_alpha).log_likelihood

    def backend_reset_test_score(self, test_rows: npt.ArrayLike) -> float:
        """Explicitly diagnostic reset score; never the OOS continuation metric."""
        if self._model is None:
            raise RuntimeError("no fitted or reconstructed backend model is available")
        values = _rows(test_rows, len(self.feature_order))
        score = float(self._model.score(values))
        if not isfinite(score):
            raise ValueError("backend TEST reset score must be finite")
        return score


class HmmlearnGMMHMMAdapter(HmmlearnGaussianHMMAdapter):
    """hmmlearn full-covariance GMM-HMM with exactly two mixtures per state."""

    def _new_model(self, state_count: int, seed: int) -> GMMHMM:
        if state_count not in (2, 3, 4, 5):
            raise ValueError("GMM-HMM candidate must be K=2, K=3, K=4, or K=5")
        settings = self.settings
        return GMMHMM(
            n_components=state_count,
            n_mix=2,
            covariance_type=settings.covariance_type,
            min_covar=settings.min_covar,
            startprob_prior=settings.startprob_prior,
            transmat_prior=settings.transmat_prior,
            weights_prior=1.0,
            means_prior=settings.means_prior,
            means_weight=settings.means_weight,
            covars_prior=settings.covars_prior,
            covars_weight=settings.covars_weight,
            algorithm="viterbi",
            random_state=seed,
            n_iter=settings.n_iter,
            tol=settings.tol,
            params="stmcw",
            verbose=False,
            implementation=settings.implementation,
            init_params="stmcw",
        )

    def _extract_model(self, model: GaussianHMM | GMMHMM) -> GaussianHMMArtifact:
        if not isinstance(model, GMMHMM):
            raise TypeError("GMM adapter requires a GMMHMM backend model")
        weights = np.asarray(model.weights_, dtype=np.float64)
        mixture_means = np.asarray(model.means_, dtype=np.float64)
        mixture_covariances = np.asarray(model.covars_, dtype=np.float64)
        state_means = np.einsum("km,kmd->kd", weights, mixture_means)
        state_covariances = np.empty(
            (model.n_components, len(self.feature_order), len(self.feature_order)), dtype=np.float64
        )
        for state in range(model.n_components):
            covariance = np.zeros_like(state_covariances[state])
            for mixture in range(2):
                delta = mixture_means[state, mixture] - state_means[state]
                covariance += weights[state, mixture] * (
                    mixture_covariances[state, mixture] + np.outer(delta, delta)
                )
            state_covariances[state] = covariance
        artifact = GaussianHMMArtifact(
            state_count=int(model.n_components),
            feature_order=self.feature_order,
            start_probabilities=tuple(float(value) for value in model.startprob_),
            transition_matrix=tuple(
                tuple(float(value) for value in row) for row in model.transmat_
            ),
            means=tuple(tuple(float(value) for value in row) for row in state_means),
            full_covariances=tuple(
                tuple(tuple(float(value) for value in row) for row in matrix)
                for matrix in state_covariances
            ),
            model_family="gmm_hmm",
            mixture_weights=tuple(tuple(float(value) for value in row) for row in weights),
            mixture_means=tuple(
                tuple(tuple(float(value) for value in mixture) for mixture in state)
                for state in mixture_means
            ),
            mixture_full_covariances=tuple(
                tuple(
                    tuple(tuple(float(value) for value in row) for row in mixture)
                    for mixture in state
                )
                for state in mixture_covariances
            ),
        )
        _validate_positive_definite(artifact)
        return artifact

    def reconstruct(self, artifact: GaussianHMMArtifact) -> None:
        if artifact.model_family != "gmm_hmm":
            raise ValueError("GMM adapter requires a GMM-HMM artifact")
        if artifact.feature_order != self.feature_order:
            raise ValueError("artifact feature order does not match adapter feature order")
        assert artifact.mixture_weights is not None
        assert artifact.mixture_means is not None
        assert artifact.mixture_full_covariances is not None
        _validate_mixture_covariances(artifact)
        model = self._new_model(artifact.state_count, seed=0)
        model.startprob_ = np.asarray(artifact.start_probabilities, dtype=np.float64)
        model.transmat_ = np.asarray(artifact.transition_matrix, dtype=np.float64)
        model.weights_ = np.asarray(artifact.mixture_weights, dtype=np.float64)
        model.means_ = np.asarray(artifact.mixture_means, dtype=np.float64)
        model.covars_ = np.asarray(artifact.mixture_full_covariances, dtype=np.float64)
        model.n_features = artifact.feature_dimension
        self._model = model
        self._artifact = artifact
