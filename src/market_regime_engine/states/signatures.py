"""Persistent-state signatures for full-covariance Gaussian HMMs."""

from __future__ import annotations

from math import isfinite, log, sqrt

from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.preprocessing.scaling import StandardScalerArtifact

StateSignature = tuple[float, ...]


def transform_emission_to_alignment_coordinate(
    mean: tuple[float, ...],
    covariance: tuple[tuple[float, ...], ...],
    fold_scaler: StandardScalerArtifact,
    reference_scaler: StandardScalerArtifact,
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    """Express a fold-local standardized emission in the fixed reference coordinate."""

    dimension = len(mean)
    if fold_scaler.feature_order != reference_scaler.feature_order:
        raise ValueError("fold and reference scalers must use the same feature order")
    if dimension != len(fold_scaler.feature_order) or any(
        len(row) != dimension for row in covariance
    ) or len(covariance) != dimension:
        raise ValueError("emission dimensions must match the scaler feature order")
    if any(not isfinite(value) for value in (*mean, *(value for row in covariance for value in row))):
        raise ValueError("emission mean and covariance must be finite")
    scale_ratios = tuple(
        fold_scale / reference_scale
        for fold_scale, reference_scale in zip(
            fold_scaler.scales, reference_scaler.scales, strict=True
        )
    )
    transformed_mean = tuple(
        (fold_mean + fold_scale * value - reference_mean) / reference_scale
        for value, fold_mean, fold_scale, reference_mean, reference_scale in zip(
            mean,
            fold_scaler.means,
            fold_scaler.scales,
            reference_scaler.means,
            reference_scaler.scales,
            strict=True,
        )
    )
    transformed_covariance = tuple(
        tuple(scale_ratios[row] * covariance[row][column] * scale_ratios[column] for column in range(dimension))
        for row in range(dimension)
    )
    return transformed_mean, transformed_covariance


def _state_signature(
    mean: tuple[float, ...], covariance: tuple[tuple[float, ...], ...]
) -> StateSignature:
    dimension = len(mean)
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
    return signature


def state_signatures(artifact: GaussianHMMArtifact) -> tuple[StateSignature, ...]:
    """Build the exact EVALUATION signature for every fitted state."""

    return tuple(
        _state_signature(
            artifact.means[state_index], artifact.distribution_covariances()[state_index]
        )
        for state_index in range(artifact.state_count)
    )


def state_signatures_in_alignment_coordinate(
    artifact: GaussianHMMArtifact,
    fold_scaler: StandardScalerArtifact,
    reference_scaler: StandardScalerArtifact,
) -> tuple[StateSignature, ...]:
    """Build signatures after transforming each fold-local emission to one fixed coordinate."""

    return tuple(
        _state_signature(
            *transform_emission_to_alignment_coordinate(
                artifact.means[state_index],
                artifact.distribution_covariances()[state_index],
                fold_scaler,
                reference_scaler,
            )
        )
        for state_index in range(artifact.state_count)
    )


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
