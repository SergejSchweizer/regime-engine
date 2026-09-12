from __future__ import annotations

import importlib.util
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from math import lgamma
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from market_regime_engine.inference.filtering import causal_filter
from market_regime_engine.models.artifacts import GaussianHMMArtifact


def _module() -> Any:
    path = Path(__file__).parents[2] / "scripts" / "verify_xetra_v4_math.py"
    spec = importlib.util.spec_from_file_location("verify_xetra_v4_math", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load independent math verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_independent_feature_score_and_soft_nmi_are_recomputable() -> None:
    module = _module()
    timestamps = tuple(
        datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=index) for index in range(10)
    )
    probabilities = tuple((1.0, 0.0) if index < 5 else (0.0, 1.0) for index in range(10))
    information_ratio, eta_squared = module.independent_feature_score(
        np.arange(10, dtype=float),
        np.asarray(timestamps, dtype=object),
        timestamps,
        probabilities,
    )

    assert information_ratio == pytest.approx(1.0)
    assert eta_squared == pytest.approx(0.7575757575757576)
    assert module.independent_soft_nmi(timestamps, probabilities, timestamps, probabilities) == 1.0


def test_independent_gaussian_likelihood_uses_forward_scaling() -> None:
    module = _module()
    actual = module.independent_gaussian_log_likelihood(
        np.asarray([[0.0], [1.0]], dtype=float),
        np.asarray([1.0]),
        np.asarray([[1.0]]),
        np.asarray([[0.0]]),
        np.asarray([[[1.0]]]),
    )

    expected = -0.5 * (2.0 * np.log(2.0 * np.pi) + 1.0)
    assert actual == pytest.approx(expected)


def test_independent_gmm_and_student_t_likelihoods_are_supported() -> None:
    module = _module()
    observations = np.asarray([[0.0], [1.0]], dtype=float)
    starts = np.asarray([1.0], dtype=float)
    transitions = np.asarray([[1.0]], dtype=float)
    gmm = module.independent_gmm_log_likelihood(
        observations,
        starts,
        transitions,
        np.asarray([[0.25, 0.75]], dtype=float),
        np.asarray([[[0.0], [0.0]]], dtype=float),
        np.asarray([[[[1.0]], [[1.0]]]], dtype=float),
    )
    expected_gaussian = -0.5 * (2.0 * np.log(2.0 * np.pi) + 1.0)
    assert gmm == pytest.approx(expected_gaussian)

    student = module.independent_student_t_log_likelihood(
        observations,
        starts,
        transitions,
        np.asarray([[0.0]], dtype=float),
        np.asarray([[[1.0]]], dtype=float),
        np.asarray([5.0], dtype=float),
    )
    expected_student = sum(
        lgamma(3.0) - lgamma(2.5) - 0.5 * np.log(5.0 * np.pi) - 3.0 * np.log1p(value * value / 5.0)
        for value in (0.0, 1.0)
    )
    assert student == pytest.approx(expected_student)


def _parity_artifact(model_family: str) -> GaussianHMMArtifact:
    base = GaussianHMMArtifact(
        state_count=2,
        feature_order=("feature_a", "feature_b"),
        start_probabilities=(0.6, 0.4),
        transition_matrix=((0.8, 0.2), (0.25, 0.75)),
        means=((0.0, 0.5), (1.0, -0.5)),
        full_covariances=(
            ((1.2, 0.15), (0.15, 0.9)),
            ((0.8, -0.1), (-0.1, 1.1)),
        ),
    )
    if model_family == "gmm_hmm":
        return replace(
            base,
            model_family="gmm_hmm",
            mixture_weights=((0.35, 0.65), (0.6, 0.4)),
            mixture_means=(
                ((-0.1, 0.45), (0.1, 0.55)),
                ((0.9, -0.55), (1.1, -0.45)),
            ),
            mixture_full_covariances=(
                (
                    ((1.1, 0.1), (0.1, 0.85)),
                    ((1.3, 0.2), (0.2, 0.95)),
                ),
                (
                    ((0.75, -0.05), (-0.05, 1.0)),
                    ((0.9, -0.12), (-0.12, 1.2)),
                ),
            ),
        )
    if model_family == "student_t_hmm":
        return replace(base, model_family="student_t_hmm", degrees_of_freedom=(5.5, 9.0))
    if model_family == "gaussian_hmm":
        return base
    raise AssertionError(f"unsupported test model family: {model_family}")


def _likelihood_payload(
    artifact: GaussianHMMArtifact,
    observations: np.ndarray,
    start_probabilities: tuple[float, ...] | np.ndarray,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "model_family": artifact.model_family,
        "observations": observations,
        "start_probabilities": np.asarray(start_probabilities, dtype=float),
        "transition_matrix": np.asarray(artifact.transition_matrix, dtype=float),
        "means": np.asarray(artifact.means, dtype=float),
        "covariances": np.asarray(artifact.full_covariances, dtype=float),
    }
    if artifact.model_family == "gmm_hmm":
        assert artifact.mixture_weights is not None
        assert artifact.mixture_means is not None
        assert artifact.mixture_full_covariances is not None
        payload.update(
            {
                "mixture_weights": np.asarray(artifact.mixture_weights, dtype=float),
                "mixture_means": np.asarray(artifact.mixture_means, dtype=float),
                "mixture_covariances": np.asarray(artifact.mixture_full_covariances, dtype=float),
            }
        )
    elif artifact.model_family == "student_t_hmm":
        assert artifact.degrees_of_freedom is not None
        payload["degrees_of_freedom"] = np.asarray(artifact.degrees_of_freedom, dtype=float)
    return payload


@pytest.mark.parametrize("model_family", ("gaussian_hmm", "gmm_hmm", "student_t_hmm"))
def test_production_train_and_causal_oos_likelihoods_match_independent_verifier(
    model_family: str,
) -> None:
    """Prove all selected v4 emission families against the standalone verifier."""

    module = _module()
    artifact = _parity_artifact(model_family)
    train = np.asarray(
        [[-0.2, 0.4], [0.1, 0.7], [0.8, -0.2], [1.2, -0.7], [0.4, 0.0]],
        dtype=float,
    )
    test = np.asarray([[0.2, 0.3], [0.9, -0.4], [1.1, -0.8]], dtype=float)

    train_filter = causal_filter(train, artifact)
    expected_train = module.independent_hmm_log_likelihood(
        _likelihood_payload(artifact, train, artifact.start_probabilities)
    )
    assert train_filter.log_likelihood == pytest.approx(expected_train, abs=1.0e-12)

    continuation_start = np.asarray(train_filter.terminal_probabilities, dtype=float) @ np.asarray(
        artifact.transition_matrix, dtype=float
    )
    oos_filter = causal_filter(
        test,
        artifact,
        initial_filtered_probabilities=train_filter.terminal_probabilities,
    )
    expected_oos = module.independent_hmm_log_likelihood(
        _likelihood_payload(artifact, test, continuation_start)
    )
    assert oos_filter.log_likelihood == pytest.approx(expected_oos, abs=1.0e-12)
