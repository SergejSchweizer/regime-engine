#!/usr/bin/env python3
"""Independently audit persisted v4 mathematical evidence.

This verifier intentionally has no imports from ``market_regime_engine``.  It
reads an Arrow snapshot plus a JSON expectation bundle emitted by an audit
run, then recomputes rank distance, state-information ratio, soft NMI, and
average-linkage silhouettes from numerical primitives.
"""

from __future__ import annotations

import argparse
import json
from math import fsum, log
from pathlib import Path

import numpy as np
import pyarrow.ipc as ipc
from scipy.stats import rankdata
from sklearn.metrics import silhouette_samples


def _read_snapshot(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        table = ipc.open_file(handle).read_all()
    return {
        name: np.asarray(table[name].to_numpy(zero_copy_only=False), dtype=object)
        for name in table.column_names
    }


def _complete(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.array(
        [
            value_left is not None
            and value_right is not None
            and np.isfinite(float(value_left))
            and np.isfinite(float(value_right))
            for value_left, value_right in zip(left, right, strict=True)
        ],
        dtype=bool,
    )
    return left[mask].astype(float), right[mask].astype(float)


def _timestamp_key(value: object) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def independent_distance(columns: dict[str, np.ndarray], features: tuple[str, ...]) -> np.ndarray:
    result = np.zeros((len(features), len(features)), dtype=float)
    for left_index, left_name in enumerate(features):
        for right_index in range(left_index):
            left, right = _complete(columns[left_name], columns[features[right_index]])
            left_rank = rankdata(left, method="average")
            right_rank = rankdata(right, method="average")
            correlation = float(np.corrcoef(left_rank, right_rank)[0, 1])
            result[left_index, right_index] = result[right_index, left_index] = 1.0 - abs(
                correlation
            )
    return result


def independent_information_ratio(
    values: np.ndarray,
    timestamps: np.ndarray,
    teacher_timestamps: tuple[object, ...],
    teacher_probabilities: tuple[tuple[float, ...], ...],
) -> float:
    by_timestamp = {
        _timestamp_key(timestamp): value
        for timestamp, value in zip(timestamps, values, strict=True)
    }
    supported = tuple(
        (float(by_timestamp[_timestamp_key(timestamp)]), probability)
        for timestamp, probability in zip(teacher_timestamps, teacher_probabilities, strict=True)
        if _timestamp_key(timestamp) in by_timestamp
        and by_timestamp[_timestamp_key(timestamp)] is not None
        and np.isfinite(float(by_timestamp[_timestamp_key(timestamp)]))
    )
    if not supported:
        raise ValueError("feature information audit has no complete teacher support")
    ranks = rankdata(tuple(value for value, _probability in supported), method="average")
    bins = tuple(min(9, int(10 * (rank - 1.0) / len(supported))) for rank in ranks)
    state_count = len(supported[0][1])
    joint = np.zeros((10, state_count), dtype=float)
    for bin_index, (_value, probability) in zip(bins, supported, strict=True):
        joint[bin_index] += np.asarray(probability, dtype=float) / len(supported)
    bin_marginal = joint.sum(axis=1)
    state_marginal = joint.sum(axis=0)
    mutual_information = fsum(
        float(value) * log(float(value) / float(bin_marginal[b] * state_marginal[k]))
        for b, row in enumerate(joint)
        for k, value in enumerate(row)
        if value > 0.0
    )
    entropy = -fsum(float(value) * log(float(value)) for value in state_marginal if value > 0.0)
    return mutual_information / entropy


def independent_feature_score(
    values: np.ndarray,
    timestamps: np.ndarray,
    teacher_timestamps: tuple[object, ...],
    teacher_probabilities: tuple[tuple[float, ...], ...],
) -> tuple[float, float]:
    """Recompute the v4 information ratio and eta-squared diagnostic."""

    by_timestamp = {
        _timestamp_key(timestamp): value
        for timestamp, value in zip(timestamps, values, strict=True)
    }
    supported = tuple(
        (float(by_timestamp[_timestamp_key(timestamp)]), probability)
        for timestamp, probability in zip(teacher_timestamps, teacher_probabilities, strict=True)
        if _timestamp_key(timestamp) in by_timestamp
        and by_timestamp[_timestamp_key(timestamp)] is not None
        and np.isfinite(float(by_timestamp[_timestamp_key(timestamp)]))
    )
    if not supported:
        raise ValueError("feature score audit has no complete support")
    numeric = np.asarray(tuple(value for value, _probability in supported), dtype=float)
    ranks = rankdata(numeric, method="average")
    bins = tuple(min(9, int(10 * (rank - 1.0) / len(supported))) for rank in ranks)
    state_count = len(supported[0][1])
    joint = np.zeros((10, state_count), dtype=float)
    for bin_index, (_value, probability) in zip(bins, supported, strict=True):
        joint[bin_index] += np.asarray(probability, dtype=float) / len(supported)
    bin_marginal = joint.sum(axis=1)
    state_marginal = joint.sum(axis=0)
    mutual_information = fsum(
        float(value) * log(float(value) / float(bin_marginal[b] * state_marginal[k]))
        for b, row in enumerate(joint)
        for k, value in enumerate(row)
        if value > 0.0
    )
    entropy = -fsum(float(value) * log(float(value)) for value in state_marginal if value > 0.0)
    overall_mean = float(numeric.mean())
    between = 0.0
    within = 0.0
    for state in range(state_count):
        weights = np.asarray([probability[state] for _value, probability in supported], dtype=float)
        denominator = float(weights.sum())
        if denominator <= 0.0:
            continue
        mean = float(np.dot(weights, numeric) / denominator)
        variance = float(np.dot(weights, (numeric - mean) ** 2) / denominator)
        fraction = denominator / len(supported)
        between += fraction * (mean - overall_mean) ** 2
        within += fraction * variance
    total = between + within
    if entropy <= 0.0 or total <= 0.0:
        raise ValueError("feature score audit has degenerate entropy or variance")
    return mutual_information / entropy, between / total


def independent_soft_nmi(
    left_timestamps: tuple[object, ...],
    left_probabilities: tuple[tuple[float, ...], ...],
    right_timestamps: tuple[object, ...],
    right_probabilities: tuple[tuple[float, ...], ...],
) -> float:
    left = {
        _timestamp_key(timestamp): probabilities
        for timestamp, probabilities in zip(left_timestamps, left_probabilities, strict=True)
    }
    right = {
        _timestamp_key(timestamp): probabilities
        for timestamp, probabilities in zip(right_timestamps, right_probabilities, strict=True)
    }
    shared = tuple(sorted(set(left) & set(right)))
    joint = np.asarray(
        [
            [
                fsum(
                    left[timestamp][left_state] * right[timestamp][right_state]
                    for timestamp in shared
                )
                / len(shared)
                for right_state in range(len(right[shared[0]]))
            ]
            for left_state in range(len(left[shared[0]]))
        ],
        dtype=float,
    )
    left_marginal = joint.sum(axis=1)
    right_marginal = joint.sum(axis=0)
    mutual_information = fsum(
        float(value) * log(float(value) / float(left_marginal[i] * right_marginal[j]))
        for i, row in enumerate(joint)
        for j, value in enumerate(row)
        if value > 0.0
    )
    left_entropy = -fsum(float(value) * log(float(value)) for value in left_marginal if value > 0.0)
    right_entropy = -fsum(
        float(value) * log(float(value)) for value in right_marginal if value > 0.0
    )
    return 2.0 * mutual_information / (left_entropy + right_entropy)


def independent_gaussian_log_likelihood(
    observations: np.ndarray,
    start_probabilities: np.ndarray,
    transition_matrix: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
) -> float:
    """Compute a full-covariance Gaussian-HMM likelihood from primitive arrays."""

    values = np.asarray(observations, dtype=float)
    starts = np.asarray(start_probabilities, dtype=float)
    transitions = np.asarray(transition_matrix, dtype=float)
    state_means = np.asarray(means, dtype=float)
    state_covariances = np.asarray(covariances, dtype=float)
    if values.ndim != 2 or starts.ndim != 1 or transitions.shape != (len(starts), len(starts)):
        raise ValueError("Gaussian likelihood primitive dimensions are invalid")
    if state_means.shape != (len(starts), values.shape[1]):
        raise ValueError("Gaussian mean dimensions are invalid")
    if state_covariances.shape != (len(starts), values.shape[1], values.shape[1]):
        raise ValueError("Gaussian covariance dimensions are invalid")
    dimension = values.shape[1]
    log_emissions = np.empty((len(values), len(starts)), dtype=float)
    normalizer = dimension * log(2.0 * np.pi)
    for state, (mean, covariance) in enumerate(zip(state_means, state_covariances, strict=True)):
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0.0 or not np.isfinite(logdet):
            raise ValueError("Gaussian covariance is not positive definite")
        centered = values - mean
        quadratic = np.einsum("ij,jk,ik->i", centered, np.linalg.inv(covariance), centered)
        log_emissions[:, state] = -0.5 * (normalizer + logdet + quadratic)
    alpha = starts * np.exp(log_emissions[0])
    total = 0.0
    scale = float(alpha.sum())
    if scale <= 0.0 or not np.isfinite(scale):
        raise ValueError("Gaussian likelihood has invalid initial scale")
    total += log(scale)
    alpha /= scale
    for row in log_emissions[1:]:
        alpha = (alpha @ transitions) * np.exp(row)
        scale = float(alpha.sum())
        if scale <= 0.0 or not np.isfinite(scale):
            raise ValueError("Gaussian likelihood has invalid forward scale")
        total += log(scale)
        alpha /= scale
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--expectations",
        type=Path,
        required=True,
        help="JSON containing independently auditable expected primitive outputs",
    )
    args = parser.parse_args()
    columns = _read_snapshot(args.snapshot)
    expected = json.loads(args.expectations.read_text(encoding="utf-8"))
    features = tuple(expected["feature_order"])
    distance = independent_distance(columns, features)
    expected_distance = np.asarray(expected["distance"], dtype=float)
    distance_error = float(np.max(np.abs(distance - expected_distance)))
    if distance_error > 1.0e-10:
        raise SystemExit(f"distance audit failed: max_abs_error={distance_error:.12g}")

    for cluster in expected.get("silhouette_clusters", []):
        labels = np.asarray(cluster["labels"], dtype=int)
        samples = silhouette_samples(distance, labels, metric="precomputed")
        actual = float(np.mean(np.where(labels == -1, 0.0, samples)))
        if abs(actual - float(cluster["mean"])) > 1.0e-10:
            raise SystemExit("silhouette audit failed")

    feature_score_errors: list[float] = []
    for item in expected.get("feature_scores", []):
        feature = str(item["feature"])
        information_ratio, eta_squared = independent_feature_score(
            columns[feature],
            columns[expected["timestamp_column"]],
            tuple(item["teacher_timestamps"]),
            tuple(tuple(row) for row in item["teacher_probabilities"]),
        )
        feature_score_errors.extend(
            (
                abs(information_ratio - float(item["state_information_ratio"])),
                abs(eta_squared - float(item["eta_squared"])),
            )
        )
    nmi_errors: list[float] = []
    for item in expected.get("prefix_nmi", []):
        actual = independent_soft_nmi(
            tuple(item["candidate_timestamps"]),
            tuple(tuple(row) for row in item["candidate_probabilities"]),
            tuple(item["teacher_timestamps"]),
            tuple(tuple(row) for row in item["teacher_probabilities"]),
        )
        nmi_errors.append(abs(actual - float(item["soft_regime_nmi"])))
    likelihood_errors: list[float] = []
    for item in expected.get("gaussian_likelihoods", []):
        actual = independent_gaussian_log_likelihood(
            np.asarray(item["observations"], dtype=float),
            np.asarray(item["start_probabilities"], dtype=float),
            np.asarray(item["transition_matrix"], dtype=float),
            np.asarray(item["means"], dtype=float),
            np.asarray(item["covariances"], dtype=float),
        )
        likelihood_errors.append(abs(actual - float(item["log_likelihood"])))
    maximum_feature_score_error = max(feature_score_errors, default=0.0)
    maximum_nmi_error = max(nmi_errors, default=0.0)
    maximum_likelihood_error = max(likelihood_errors, default=0.0)
    if maximum_feature_score_error > 1.0e-10:
        raise SystemExit("feature-score audit failed")
    if maximum_nmi_error > 1.0e-10:
        raise SystemExit("soft-NMI audit failed")
    if maximum_likelihood_error > 1.0e-10:
        raise SystemExit("Gaussian likelihood audit failed")

    print(
        json.dumps(
            {
                "distance_max_abs_error": distance_error,
                "feature_score_max_abs_error": maximum_feature_score_error,
                "soft_nmi_max_abs_error": maximum_nmi_error,
                "gaussian_likelihood_max_abs_error": maximum_likelihood_error,
                "status": "verified",
            }
        )
    )


if __name__ == "__main__":
    main()
