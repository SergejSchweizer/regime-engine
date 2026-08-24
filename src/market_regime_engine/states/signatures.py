"""Persistent-state signatures for full-covariance Gaussian HMMs."""

from __future__ import annotations

from math import isfinite, log, sqrt

from market_regime_engine.models.artifacts import GaussianHMMArtifact

StateSignature = tuple[float, ...]


def state_signatures(artifact: GaussianHMMArtifact) -> tuple[StateSignature, ...]:
    """Build the exact EVALUATION signature for every fitted state.

    Each signature concatenates the mean vector, log standard deviations, and
    the upper off-diagonal triangle of the covariance-derived correlation matrix.
    """

    signatures: list[StateSignature] = []
    dimension = artifact.feature_dimension
    for state_index in range(artifact.state_count):
        mean = artifact.means[state_index]
        covariance = artifact.full_covariances[state_index]
        variances = tuple(covariance[index][index] for index in range(dimension))
        if any(value <= 0.0 or not isfinite(value) for value in variances):
            raise ValueError("state signature requires positive finite covariance diagonal")
        log_standard_deviations = tuple(log(sqrt(value)) for value in variances)
        correlations: list[float] = []
        for row in range(dimension):
            for column in range(row + 1, dimension):
                denominator = sqrt(variances[row] * variances[column])
                correlation = covariance[row][column] / denominator
                if not isfinite(correlation):
                    raise ValueError("state signature correlation must be finite")
                correlations.append(correlation)
        signature = (*mean, *log_standard_deviations, *correlations)
        if any(not isfinite(value) for value in signature):
            raise ValueError("state signature components must all be finite")
        signatures.append(signature)
    return tuple(signatures)


def signature_sort_key(signature: StateSignature) -> tuple[float, ...]:
    """Return the pinned first-fold sort key rounded componentwise to 10 decimals."""

    if not signature or any(not isfinite(value) for value in signature):
        raise ValueError("signature sort key requires a non-empty finite signature")
    return tuple(round(value, 10) for value in signature)


def rms_distance(left: StateSignature, right: StateSignature) -> float:
    """Root-mean-square distance between equal-dimensional finite signatures."""

    if not left or len(left) != len(right):
        raise ValueError("RMS state signatures must have the same non-zero dimension")
    if any(not isfinite(value) for value in (*left, *right)):
        raise ValueError("RMS state signatures must be finite")
    mean_squared = sum((a - b) ** 2 for a, b in zip(left, right, strict=True)) / len(left)
    result = sqrt(mean_squared)
    if not isfinite(result):
        raise ValueError("RMS state-signature distance must be finite")
    return result
