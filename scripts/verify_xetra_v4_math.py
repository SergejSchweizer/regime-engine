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
import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from math import fsum, log
from pathlib import Path
from typing import cast

import numpy as np
import pyarrow.ipc as ipc
from scipy.special import gammaln, logsumexp
from scipy.stats import rankdata
from sklearn.metrics import silhouette_samples

_PROBABILITY_TOLERANCE = 1.0e-10
_AUDIT_TOLERANCE = 1.0e-10
_AUDIT_COLUMNS: dict[str, np.ndarray] | None = None
_AUDIT_THREAD_LIMITER: object | None = None


def _initialize_audit_worker(columns: dict[str, np.ndarray]) -> None:
    """Install immutable audit inputs and one native numerical lane per worker."""

    global _AUDIT_COLUMNS, _AUDIT_THREAD_LIMITER
    _AUDIT_COLUMNS = columns
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    try:
        from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]
    except ImportError:
        _AUDIT_THREAD_LIMITER = None
    else:
        limiter = threadpool_limits(limits=1)
        limiter.__enter__()
        _AUDIT_THREAD_LIMITER = limiter


def _audit_worker_count(requested: int | None, task_count: int) -> int:
    if task_count < 1:
        return 1
    if requested is not None and (isinstance(requested, bool) or requested < 1):
        raise ValueError("workers must be positive")
    available = _audit_available_cpu_count()
    return min(requested or available, available, task_count)


def _audit_requested_worker_limit(requested: int | None) -> int:
    """Return the requested process limit before it is bounded by task count."""

    if requested is not None and (isinstance(requested, bool) or requested < 1):
        raise ValueError("workers must be positive")
    return min(requested or _audit_available_cpu_count(), _audit_available_cpu_count())


def _audit_affinity_cpu_count() -> int:
    """Return CPUs allowed by the process affinity mask, with a portable fallback."""

    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            return max(1, len(affinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def _audit_cgroup_cpu_limit(
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    membership_path: Path = Path("/proc/self/cgroup"),
) -> int | None:
    """Read hard CPU quotas from the active cgroup, independently of production code.

    Both cgroup v2 ``cpu.max`` and cgroup v1 ``cpu.cfs_*`` are supported.  A
    missing, unlimited, or malformed quota is ignored; an integer worker count
    is conservative for fractional quotas (for example, 1.5 CPUs permits one
    CPU-bound process).
    """

    roots = [cgroup_root]
    try:
        membership = membership_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        membership = []
    for line in membership:
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        _hierarchy, controllers, relative_path = fields
        if controllers == "":
            roots.append(cgroup_root / relative_path.lstrip("/"))
        elif "cpu" in controllers.split(","):
            roots.append(cgroup_root / "cpu" / relative_path.lstrip("/"))

    limits: list[int] = []
    for root in dict.fromkeys(roots):
        try:
            cpu_max = (root / "cpu.max").read_text(encoding="utf-8").split()
        except OSError:
            cpu_max = []
        if cpu_max and cpu_max[0] != "max":
            try:
                quota = int(cpu_max[0])
                period = int(cpu_max[1]) if len(cpu_max) > 1 else 100_000
            except ValueError:
                quota = period = 0
            if quota > 0 and period > 0:
                limits.append(max(1, math.floor(quota / period)))

        try:
            quota = int((root / "cpu.cfs_quota_us").read_text(encoding="utf-8").strip())
            period = int((root / "cpu.cfs_period_us").read_text(encoding="utf-8").strip())
        except OSError, ValueError:
            continue
        if quota > 0 and period > 0:
            limits.append(max(1, math.floor(quota / period)))
    return min(limits) if limits else None


def _audit_available_cpu_count() -> int:
    """Return the CPUs usable by this verifier under affinity and cgroup limits."""

    affinity_count = _audit_affinity_cpu_count()
    quota = _audit_cgroup_cpu_limit()
    return max(1, min(affinity_count, quota)) if quota is not None else affinity_count


def _audit_process_context() -> multiprocessing.context.BaseContext:
    methods = multiprocessing.get_all_start_methods()
    if "fork" in methods:
        return multiprocessing.get_context("fork")
    return multiprocessing.get_context("spawn")


def _read_snapshot(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise ValueError(f"snapshot is not a regular file: {path}")
    with path.open("rb") as handle:
        table = ipc.open_file(handle).read_all()
    if table.num_rows < 1 or not table.column_names:
        raise ValueError("snapshot must contain at least one row and one column")
    columns = {
        name: np.asarray(table[name].to_numpy(zero_copy_only=False), dtype=object)
        for name in table.column_names
    }
    row_counts = {len(values) for values in columns.values()}
    if row_counts != {table.num_rows}:
        raise ValueError("snapshot columns must have one common row count")
    for values in columns.values():
        values.setflags(write=False)
    return columns


def _complete(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_values: list[float] = []
    right_values: list[float] = []
    for value_left, value_right in zip(left, right, strict=True):
        for value, name in ((value_left, "left"), (value_right, "right")):
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"rank distance {name} value is not numeric") from error
            if not math.isfinite(numeric):
                raise ValueError(f"rank distance {name} value is nonfinite")
        if value_left is not None and value_right is not None:
            left_values.append(float(value_left))
            right_values.append(float(value_right))
    return np.asarray(left_values, dtype=float), np.asarray(right_values, dtype=float)


def _rank_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Compute one finite rank distance or reject an unverifiable pair."""

    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape or len(left) < 2:
        raise ValueError("rank distance requires at least two complete observations")
    left_rank = rankdata(left, method="average")
    right_rank = rankdata(right, method="average")
    correlation = float(np.corrcoef(left_rank, right_rank)[0, 1])
    if not math.isfinite(correlation) or not -1.0 <= correlation <= 1.0:
        raise ValueError("rank distance correlation is nonfinite or outside [-1,1]")
    distance = 1.0 - abs(correlation)
    if not math.isfinite(distance) or not 0.0 <= distance <= 1.0:
        raise ValueError("rank distance is nonfinite or outside [0,1]")
    return distance


def _timestamp_key(value: object) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _timestamp_mapping(
    timestamps: tuple[object, ...],
    values: tuple[object, ...],
    name: str,
) -> dict[str, object]:
    if not timestamps or len(timestamps) != len(values):
        raise ValueError(f"{name} timestamps and values must be non-empty and aligned")
    keys = tuple(_timestamp_key(value) for value in timestamps)
    if len(set(keys)) != len(keys):
        raise ValueError(f"{name} timestamps must be unique")
    return dict(zip(keys, values, strict=True))


def _finite_scalar(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _probability_matrix(value: object, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a rectangular numeric matrix") from error
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    if not np.allclose(
        matrix.sum(axis=1),
        1.0,
        rtol=0.0,
        atol=_PROBABILITY_TOLERANCE,
    ):
        raise ValueError(f"{name} rows must be normalized")
    return matrix


def _probability_vector(value: object, state_count: int, name: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric vector") from error
    if vector.shape != (state_count,):
        raise ValueError(f"{name} has invalid dimensions")
    if not np.all(np.isfinite(vector)) or np.any(vector < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    if not np.isclose(
        float(vector.sum()),
        1.0,
        rtol=0.0,
        atol=_PROBABILITY_TOLERANCE,
    ):
        raise ValueError(f"{name} must be normalized")
    return vector


def _aligned_probability_matrix(
    timestamps: object,
    probabilities: object,
    name: str,
) -> tuple[tuple[object, ...], np.ndarray, dict[str, np.ndarray]]:
    if not isinstance(timestamps, (list, tuple)):
        raise ValueError(f"{name} timestamps must be a list")
    timestamp_values = tuple(timestamps)
    matrix = _probability_matrix(probabilities, f"{name} probabilities")
    if len(timestamp_values) != matrix.shape[0]:
        raise ValueError(f"{name} timestamps and probabilities are not aligned")
    raw_mapping = _timestamp_mapping(
        timestamp_values,
        tuple(row for row in matrix),
        name,
    )
    return (
        timestamp_values,
        matrix,
        {key: np.asarray(row, dtype=float) for key, row in raw_mapping.items()},
    )


def independent_distance(columns: dict[str, np.ndarray], features: tuple[str, ...]) -> np.ndarray:
    if not features or len(set(features)) != len(features):
        raise ValueError("distance feature order must be non-empty and duplicate-free")
    if any(feature not in columns for feature in features):
        raise ValueError("distance feature order references a missing snapshot column")
    result = np.zeros((len(features), len(features)), dtype=float)
    for left_index, left_name in enumerate(features):
        for right_index in range(left_index):
            left, right = _complete(columns[left_name], columns[features[right_index]])
            result[left_index, right_index] = result[right_index, left_index] = _rank_distance(
                left, right
            )
    return result


def _distance_chunk(
    columns: dict[str, np.ndarray],
    features: tuple[str, ...],
    pairs: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int, float], ...]:
    values: list[tuple[int, int, float]] = []
    for left_index, right_index in pairs:
        left, right = _complete(columns[features[left_index]], columns[features[right_index]])
        values.append((left_index, right_index, _rank_distance(left, right)))
    return tuple(values)


def _process_distance_chunk(
    task: tuple[int, tuple[str, ...], tuple[tuple[int, int], ...]],
) -> tuple[int, tuple[tuple[int, int, float], ...]]:
    if _AUDIT_COLUMNS is None:
        raise RuntimeError("audit process worker was not initialized")
    dossier_index, features, pairs = task
    return dossier_index, _distance_chunk(_AUDIT_COLUMNS, features, pairs)


def _silhouette_value(
    distance: np.ndarray,
    labels: np.ndarray,
    cluster_count: int,
) -> float:
    """Compute one deterministic silhouette cut after strict input checks."""

    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("silhouette distance matrix must be square")
    if not np.all(np.isfinite(distance)):
        raise ValueError("silhouette distance matrix must be finite")
    if not np.array_equal(distance, distance.T) or not np.all(np.diag(distance) == 0.0):
        raise ValueError("silhouette distance matrix must be symmetric with zero diagonal")
    if labels.shape != (distance.shape[0],):
        raise ValueError("silhouette labels have invalid dimensions")
    if cluster_count < 2 or cluster_count >= len(labels):
        raise ValueError("silhouette cluster count must be between two and n-1")
    expected_labels = np.arange(cluster_count, dtype=np.int64)
    if not np.array_equal(np.unique(labels), expected_labels):
        raise ValueError("silhouette labels must be contiguous and match cluster count")
    samples = np.asarray(silhouette_samples(distance, labels, metric="precomputed"), dtype=float)
    for label in expected_labels:
        if np.count_nonzero(labels == label) == 1:
            samples[labels == label] = 0.0
    if not np.all(np.isfinite(samples)):
        raise ValueError("silhouette samples must be finite")
    mean = float(np.mean(samples))
    if not math.isfinite(mean) or not -1.0 <= mean <= 1.0:
        raise ValueError("silhouette mean is nonfinite or outside [-1,1]")
    return mean


def _process_silhouette(
    task: tuple[int, int, int, tuple[tuple[float, ...], ...], tuple[int, ...]],
) -> tuple[int, int, float]:
    dossier_index, cluster_index, cluster_count, raw_distance, raw_labels = task
    distance = np.asarray(raw_distance, dtype=float)
    labels = np.asarray(raw_labels, dtype=np.int64)
    return (
        dossier_index,
        cluster_index,
        _silhouette_value(
            distance,
            labels,
            cluster_count,
        ),
    )


def _process_feature_score(
    task: tuple[int, str, dict[str, object], str],
) -> tuple[int, float, float]:
    if _AUDIT_COLUMNS is None:
        raise RuntimeError("audit process worker was not initialized")
    item_index, feature, item, timestamp_column = task
    information_ratio, eta_squared = independent_feature_score(
        _AUDIT_COLUMNS[feature],
        _AUDIT_COLUMNS[timestamp_column],
        tuple(item["teacher_timestamps"]),
        tuple(tuple(row) for row in item["teacher_probabilities"]),
    )
    return item_index, information_ratio, eta_squared


def _process_nmi(
    task: tuple[int, dict[str, object]],
) -> tuple[int, float]:
    item_index, item = task
    return item_index, independent_soft_nmi(
        tuple(item["candidate_timestamps"]),
        tuple(tuple(row) for row in item["candidate_probabilities"]),
        tuple(item["teacher_timestamps"]),
        tuple(tuple(row) for row in item["teacher_probabilities"]),
    )


def _process_likelihood(task: tuple[int, dict[str, object]]) -> tuple[int, float]:
    item_index, item = task
    return item_index, independent_hmm_log_likelihood(item)


def independent_information_ratio(
    values: np.ndarray,
    timestamps: np.ndarray,
    teacher_timestamps: tuple[object, ...],
    teacher_probabilities: tuple[tuple[float, ...], ...],
) -> float:
    return independent_feature_score(
        values,
        timestamps,
        teacher_timestamps,
        teacher_probabilities,
    )[0]


def independent_feature_score(
    values: np.ndarray,
    timestamps: np.ndarray,
    teacher_timestamps: tuple[object, ...],
    teacher_probabilities: tuple[tuple[float, ...], ...],
) -> tuple[float, float]:
    """Recompute the v4 information ratio and eta-squared diagnostic."""

    feature_values = tuple(values)
    by_timestamp = _timestamp_mapping(tuple(timestamps), feature_values, "feature")
    teacher_time_values, teacher_matrix, teacher_by_timestamp = _aligned_probability_matrix(
        teacher_timestamps,
        teacher_probabilities,
        "teacher",
    )
    supported = tuple(
        (_finite_scalar(by_timestamp[key], "feature value"), teacher_by_timestamp[key])
        for key in (_timestamp_key(timestamp) for timestamp in teacher_time_values)
        if key in by_timestamp and by_timestamp[key] is not None
    )
    if not supported:
        raise ValueError("feature score audit has no complete support")
    numeric = np.asarray(tuple(value for value, _probability in supported), dtype=float)
    ranks = rankdata(numeric, method="average")
    bins = tuple(min(9, int(10 * (rank - 1.0) / len(supported))) for rank in ranks)
    state_count = teacher_matrix.shape[1]
    joint = np.zeros((10, state_count), dtype=float)
    for bin_index, (_value, probability) in zip(bins, supported, strict=True):
        joint[bin_index] += probability / len(supported)
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
    information_ratio = mutual_information / entropy
    eta_squared = between / total
    if not math.isfinite(information_ratio) or not math.isfinite(eta_squared):
        raise ValueError("feature score audit produced a nonfinite result")
    return information_ratio, eta_squared


def independent_soft_nmi(
    left_timestamps: tuple[object, ...],
    left_probabilities: tuple[tuple[float, ...], ...],
    right_timestamps: tuple[object, ...],
    right_probabilities: tuple[tuple[float, ...], ...],
) -> float:
    left_times, left_matrix, left = _aligned_probability_matrix(
        left_timestamps,
        left_probabilities,
        "left",
    )
    _right_times, right_matrix, right = _aligned_probability_matrix(
        right_timestamps,
        right_probabilities,
        "right",
    )
    del left_matrix, right_matrix
    shared = tuple(sorted(set(_timestamp_key(value) for value in left_times) & set(right)))
    if not shared:
        raise ValueError("soft-NMI audit requires shared timestamps")
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
    if not np.all(np.isfinite(joint)) or np.any(joint < 0.0):
        raise ValueError("soft-NMI joint distribution is invalid")
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
    entropy = left_entropy + right_entropy
    if entropy <= 0.0 or not math.isfinite(entropy):
        raise ValueError("soft-NMI audit has degenerate entropy")
    result = 2.0 * mutual_information / entropy
    if not math.isfinite(result) or not 0.0 <= result <= 1.0 + _AUDIT_TOLERANCE:
        raise ValueError("soft-NMI audit result is invalid")
    return result


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
    _validate_forward_primitives(values, starts, transitions, "Gaussian likelihood")
    state_means = np.asarray(means, dtype=float)
    state_covariances = np.asarray(covariances, dtype=float)
    if state_means.shape != (len(starts), values.shape[1]):
        raise ValueError("Gaussian mean dimensions are invalid")
    if state_covariances.shape != (len(starts), values.shape[1], values.shape[1]):
        raise ValueError("Gaussian covariance dimensions are invalid")
    if not np.all(np.isfinite(state_means)) or not np.all(np.isfinite(state_covariances)):
        raise ValueError("Gaussian means and covariances must be finite")
    dimension = values.shape[1]
    log_emissions = np.empty((len(values), len(starts)), dtype=float)
    normalizer = dimension * log(2.0 * np.pi)
    for state, (mean, covariance) in enumerate(zip(state_means, state_covariances, strict=True)):
        if not np.allclose(covariance, covariance.T, rtol=0.0, atol=_AUDIT_TOLERANCE):
            raise ValueError("Gaussian covariance must be symmetric")
        covariance = (covariance + covariance.T) / 2.0
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0.0 or not np.isfinite(logdet):
            raise ValueError("Gaussian covariance is not positive definite")
        centered = values - mean
        solved = np.linalg.solve(covariance, centered.T).T
        quadratic = np.einsum("ij,ij->i", centered, solved)
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


def _independent_gaussian_log_density(
    values: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
) -> np.ndarray:
    """Return one independent full-covariance Gaussian log density per row/state."""

    observations = np.asarray(values, dtype=float)
    state_means = np.asarray(means, dtype=float)
    state_covariances = np.asarray(covariances, dtype=float)
    if observations.ndim != 2 or state_means.ndim != 2:
        raise ValueError("density primitives must be two-dimensional")
    if observations.shape[0] < 1 or observations.shape[1] < 1:
        raise ValueError("density observations must be non-empty")
    if not np.all(np.isfinite(observations)) or not np.all(np.isfinite(state_means)):
        raise ValueError("density observations and means must be finite")
    if state_means.shape[1] != observations.shape[1]:
        raise ValueError("density feature dimensions are inconsistent")
    if state_covariances.shape != (
        state_means.shape[0],
        observations.shape[1],
        observations.shape[1],
    ):
        raise ValueError("density covariance dimensions are invalid")
    dimension = observations.shape[1]
    output = np.empty((len(observations), len(state_means)), dtype=float)
    for state, (mean, covariance) in enumerate(zip(state_means, state_covariances, strict=True)):
        if not np.all(np.isfinite(covariance)):
            raise ValueError("density covariance must be finite")
        if not np.allclose(covariance, covariance.T, rtol=0.0, atol=_AUDIT_TOLERANCE):
            raise ValueError("density covariance must be symmetric")
        symmetric = (covariance + covariance.T) / 2.0
        sign, logdet = np.linalg.slogdet(symmetric)
        if sign <= 0.0 or not np.isfinite(logdet):
            raise ValueError("density covariance must be positive definite")
        centered = observations - mean
        solved = np.linalg.solve(symmetric, centered.T).T
        quadratic = np.einsum("ij,ij->i", centered, solved)
        output[:, state] = -0.5 * (dimension * np.log(2.0 * np.pi) + logdet + quadratic)
    return output


def independent_gmm_log_likelihood(
    observations: np.ndarray,
    start_probabilities: np.ndarray,
    transition_matrix: np.ndarray,
    mixture_weights: np.ndarray,
    mixture_means: np.ndarray,
    mixture_covariances: np.ndarray,
) -> float:
    """Compute a full-covariance GMM-HMM likelihood independently."""

    values = np.asarray(observations, dtype=float)
    starts = np.asarray(start_probabilities, dtype=float)
    transitions = np.asarray(transition_matrix, dtype=float)
    weights = np.asarray(mixture_weights, dtype=float)
    means = np.asarray(mixture_means, dtype=float)
    covariances = np.asarray(mixture_covariances, dtype=float)
    state_count = starts.shape[0] if starts.ndim == 1 else 0
    _validate_forward_primitives(values, starts, transitions, "GMM likelihood")
    if transitions.shape != (state_count, state_count):
        raise ValueError("GMM transition dimensions are invalid")
    if weights.ndim != 2 or weights.shape[0] != state_count:
        raise ValueError("GMM mixture-weight dimensions are invalid")
    if means.ndim != 3 or means.shape[:2] != weights.shape:
        raise ValueError("GMM mixture-mean dimensions are invalid")
    if covariances.shape != (
        state_count,
        weights.shape[1],
        values.shape[1],
        values.shape[1],
    ):
        raise ValueError("GMM mixture-covariance dimensions are invalid")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("GMM mixture weights must be finite and non-negative")
    if not np.allclose(weights.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("GMM mixture weights must be normalized")
    if not np.all(np.isfinite(means)):
        raise ValueError("GMM mixture means must be finite")
    component_log_density = np.empty(
        (len(values), state_count, weights.shape[1]),
        dtype=float,
    )
    for state in range(state_count):
        component_log_density[:, state, :] = (
            _independent_gaussian_log_density(values, means[state], covariances[state])
            + np.log(weights[state])[None, :]
        )
    emissions = logsumexp(component_log_density, axis=2)
    return _forward_log_likelihood(emissions, starts, transitions)


def independent_student_t_log_likelihood(
    observations: np.ndarray,
    start_probabilities: np.ndarray,
    transition_matrix: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    degrees_of_freedom: np.ndarray,
) -> float:
    """Compute a full-covariance multivariate Student-t HMM likelihood independently."""

    values = np.asarray(observations, dtype=float)
    starts = np.asarray(start_probabilities, dtype=float)
    transitions = np.asarray(transition_matrix, dtype=float)
    state_means = np.asarray(means, dtype=float)
    state_covariances = np.asarray(covariances, dtype=float)
    degrees = np.asarray(degrees_of_freedom, dtype=float)
    state_count = starts.shape[0] if starts.ndim == 1 else 0
    _validate_forward_primitives(values, starts, transitions, "Student-t likelihood")
    if transitions.shape != (state_count, state_count):
        raise ValueError("Student-t transition dimensions are invalid")
    if state_means.shape != (state_count, values.shape[1]):
        raise ValueError("Student-t mean dimensions are invalid")
    if state_covariances.shape != (state_count, values.shape[1], values.shape[1]):
        raise ValueError("Student-t covariance dimensions are invalid")
    if (
        degrees.shape != (state_count,)
        or not np.all(np.isfinite(degrees))
        or np.any(degrees <= 2.0)
    ):
        raise ValueError("Student-t degrees of freedom are invalid")
    if not np.all(np.isfinite(state_means)) or not np.all(np.isfinite(state_covariances)):
        raise ValueError("Student-t means and covariances must be finite")
    emissions = np.empty((len(values), state_count), dtype=float)
    dimension = values.shape[1]
    for state, (mean, covariance, degree) in enumerate(
        zip(state_means, state_covariances, degrees, strict=True)
    ):
        if not np.allclose(covariance, covariance.T, rtol=0.0, atol=_AUDIT_TOLERANCE):
            raise ValueError("Student-t covariance must be symmetric")
        symmetric = (covariance + covariance.T) / 2.0
        sign, logdet = np.linalg.slogdet(symmetric)
        if sign <= 0.0 or not np.isfinite(logdet):
            raise ValueError("Student-t covariance must be positive definite")
        centered = values - mean
        solved = np.linalg.solve(symmetric, centered.T).T
        quadratic = np.einsum("ij,ij->i", centered, solved)
        emissions[:, state] = (
            gammaln((degree + dimension) / 2.0)
            - gammaln(degree / 2.0)
            - 0.5 * (dimension * np.log(degree * np.pi) + logdet)
            - 0.5 * (degree + dimension) * np.log1p(quadratic / degree)
        )
    return _forward_log_likelihood(emissions, starts, transitions)


def _forward_log_likelihood(
    log_emissions: np.ndarray,
    start_probabilities: np.ndarray,
    transition_matrix: np.ndarray,
) -> float:
    """Apply a scaled forward recursion to independently computed emissions."""

    starts = np.asarray(start_probabilities, dtype=float)
    transitions = np.asarray(transition_matrix, dtype=float)
    if log_emissions.ndim != 2 or log_emissions.shape[0] < 1:
        raise ValueError("forward emission dimensions are invalid")
    if not np.all(np.isfinite(log_emissions)):
        raise ValueError("forward emissions must be finite")
    if starts.ndim != 1 or log_emissions.shape[1] != starts.shape[0]:
        raise ValueError("forward emission/state dimensions are invalid")
    _validate_forward_primitives(
        np.empty((log_emissions.shape[0], 1), dtype=float),
        starts,
        transitions,
        "forward",
    )
    with np.errstate(divide="ignore"):
        alpha = np.log(starts) + log_emissions[0]
    total = float(logsumexp(alpha))
    if not np.isfinite(total):
        raise ValueError("forward likelihood has invalid initial scale")
    alpha -= total
    for row in log_emissions[1:]:
        with np.errstate(divide="ignore"):
            alpha = logsumexp(alpha[:, None] + np.log(transitions), axis=0) + row
        increment = float(logsumexp(alpha))
        if not np.isfinite(increment):
            raise ValueError("forward likelihood has invalid scale")
        total += increment
        alpha -= increment
    if not math.isfinite(total):
        raise ValueError("forward likelihood result is nonfinite")
    return total


def _validate_forward_primitives(
    observations: np.ndarray,
    starts: np.ndarray,
    transitions: np.ndarray,
    name: str,
) -> None:
    """Validate HMM dimensions and probabilities before numerical recursion."""

    if observations.ndim != 2 or observations.shape[0] < 1 or observations.shape[1] < 1:
        raise ValueError(f"{name} observations must be non-empty and two-dimensional")
    if not np.all(np.isfinite(observations)):
        raise ValueError(f"{name} observations must be finite")
    if starts.ndim != 1 or starts.shape[0] < 1:
        raise ValueError(f"{name} start probabilities have invalid dimensions")
    state_count = starts.shape[0]
    _probability_vector(starts, state_count, f"{name} start probabilities")
    if transitions.shape != (state_count, state_count):
        raise ValueError(f"{name} transition matrix has invalid dimensions")
    if not np.all(np.isfinite(transitions)) or np.any(transitions < 0.0):
        raise ValueError(f"{name} transition matrix must be finite and non-negative")
    if not np.allclose(
        transitions.sum(axis=1),
        1.0,
        rtol=0.0,
        atol=_PROBABILITY_TOLERANCE,
    ):
        raise ValueError(f"{name} transition rows must be normalized")


def independent_hmm_log_likelihood(item: dict[str, object]) -> float:
    """Dispatch one expectation record without importing production model code."""

    family = str(item.get("model_family", "gaussian_hmm"))
    common = (
        np.asarray(item["observations"], dtype=float),
        np.asarray(item["start_probabilities"], dtype=float),
        np.asarray(item["transition_matrix"], dtype=float),
    )
    if family == "gaussian_hmm":
        return independent_gaussian_log_likelihood(
            *common,
            np.asarray(item["means"], dtype=float),
            np.asarray(item["covariances"], dtype=float),
        )
    if family == "gmm_hmm":
        return independent_gmm_log_likelihood(
            *common,
            np.asarray(item["mixture_weights"], dtype=float),
            np.asarray(item["mixture_means"], dtype=float),
            np.asarray(item["mixture_covariances"], dtype=float),
        )
    if family == "student_t_hmm":
        return independent_student_t_log_likelihood(
            *common,
            np.asarray(item["means"], dtype=float),
            np.asarray(item["covariances"], dtype=float),
            np.asarray(item["degrees_of_freedom"], dtype=float),
        )
    raise ValueError(f"unsupported likelihood model family: {family}")


def _audit_dossiers(expected: object) -> tuple[dict[str, object], ...]:
    """Return dossiers after validating their deterministic outer-fold envelope."""

    if not isinstance(expected, dict):
        raise SystemExit("math expectations must be a JSON object")
    raw_dossiers = expected.get("fold_audits")
    if raw_dossiers is None:
        return (expected,)
    if not isinstance(raw_dossiers, list) or not raw_dossiers:
        raise SystemExit("fold_audits must be a non-empty list")
    dossiers = tuple(item for item in raw_dossiers if isinstance(item, dict))
    if len(dossiers) != len(raw_dossiers):
        raise SystemExit("each fold audit must be an object")
    indices: list[int] = []
    for item in dossiers:
        value = item.get("outer_fold_index")
        if isinstance(value, bool) or not isinstance(value, int):
            raise SystemExit("each fold audit must declare an integer outer_fold_index")
        indices.append(value)
    if tuple(indices) != tuple(sorted(set(indices))):
        raise SystemExit("fold audit outer-fold indices must be unique and sorted")
    declared = expected.get("audit_outer_fold_indices")
    if declared != indices:
        raise SystemExit("audit_outer_fold_indices does not match fold_audits")
    return dossiers


def _required_string(item: dict[str, object], key: str, context: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value:
        raise SystemExit(f"{context}.{key} must be a non-empty string")
    return value


def _required_list(item: dict[str, object], key: str, context: str) -> list[object]:
    value = item.get(key)
    if not isinstance(value, list):
        raise SystemExit(f"{context}.{key} must be a list")
    return value


def _finite_matrix(value: object, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise SystemExit(f"{name} must be a rectangular numeric matrix") from error
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise SystemExit(f"{name} must be a finite two-dimensional matrix")
    return matrix


def _validate_likelihood_contract(item: dict[str, object], context: str) -> None:
    family = item.get("model_family", "gaussian_hmm")
    if family not in {"gaussian_hmm", "gmm_hmm", "student_t_hmm"}:
        raise SystemExit(f"{context}.model_family is unsupported")
    for key in ("observations", "start_probabilities", "transition_matrix", "means", "covariances"):
        if key not in item:
            raise SystemExit(f"{context}.{key} is required")
    if "log_likelihood" not in item:
        raise SystemExit(f"{context}.log_likelihood is required")
    try:
        _finite_scalar(item["log_likelihood"], f"{context}.log_likelihood")
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if family == "gmm_hmm":
        for key in ("mixture_weights", "mixture_means", "mixture_covariances"):
            if key not in item:
                raise SystemExit(f"{context}.{key} is required for GMM-HMM")
    elif family == "student_t_hmm" and "degrees_of_freedom" not in item:
        raise SystemExit(f"{context}.degrees_of_freedom is required for Student-t HMM")


def _validate_dossier_contract(
    dossier: dict[str, object],
    columns: dict[str, np.ndarray],
    dossier_index: int,
) -> tuple[
    tuple[str, ...],
    str,
    np.ndarray,
    tuple[tuple[int, tuple[int, ...], float], ...],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Validate one complete dossier before scheduling any process work."""

    context = f"fold_audits[{dossier_index}]"
    raw_features = _required_list(dossier, "feature_order", context)
    if not raw_features or any(not isinstance(value, str) or not value for value in raw_features):
        raise SystemExit(f"{context}.feature_order must contain non-empty strings")
    features = tuple(cast(str, value) for value in raw_features)
    if len(set(features)) != len(features):
        raise SystemExit(f"{context}.feature_order must be duplicate-free")
    missing = tuple(feature for feature in features if feature not in columns)
    if missing:
        raise SystemExit(f"{context}.feature_order references missing columns: {missing}")
    timestamp_column = _required_string(dossier, "timestamp_column", context)
    if timestamp_column not in columns:
        raise SystemExit(f"{context}.timestamp_column references a missing column")
    expected_distance = _finite_matrix(dossier.get("distance"), f"{context}.distance")
    expected_shape = (len(features), len(features))
    if expected_distance.shape != expected_shape:
        raise SystemExit(f"{context}.distance dimensions are invalid")

    raw_silhouettes = dossier.get("silhouette_clusters", [])
    if not isinstance(raw_silhouettes, list):
        raise SystemExit(f"{context}.silhouette_clusters must be a list")
    silhouettes: list[tuple[int, tuple[int, ...], float]] = []
    seen_cluster_counts: set[int] = set()
    for cluster_index, raw_cluster in enumerate(raw_silhouettes):
        if not isinstance(raw_cluster, dict):
            raise SystemExit(f"{context}.silhouette_clusters[{cluster_index}] must be an object")
        cluster_count = raw_cluster.get("cluster_count")
        labels_value = raw_cluster.get("labels")
        if isinstance(cluster_count, bool) or not isinstance(cluster_count, int):
            raise SystemExit(
                f"{context}.silhouette_clusters[{cluster_index}].cluster_count is invalid"
            )
        if cluster_count in seen_cluster_counts:
            raise SystemExit(f"{context}.silhouette_clusters contains duplicate cluster counts")
        seen_cluster_counts.add(cluster_count)
        if not isinstance(labels_value, list) or len(labels_value) != len(features):
            raise SystemExit(
                f"{context}.silhouette_clusters[{cluster_index}].labels dimensions are invalid"
            )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in labels_value):
            raise SystemExit(f"{context}.silhouette_clusters[{cluster_index}].labels are invalid")
        labels = tuple(cast(int, value) for value in labels_value)
        if cluster_count < 2 or cluster_count >= len(features):
            raise SystemExit(
                f"{context}.silhouette_clusters[{cluster_index}].cluster_count is invalid"
            )
        if tuple(sorted(set(labels))) != tuple(range(cluster_count)):
            raise SystemExit(
                f"{context}.silhouette_clusters[{cluster_index}].labels are not contiguous"
            )
        try:
            mean = _finite_scalar(raw_cluster.get("mean"), f"{context}.silhouette mean")
        except ValueError as error:
            raise SystemExit(str(error)) from error
        silhouettes.append((cluster_count, labels, mean))

    raw_feature_scores = dossier.get("feature_scores", [])
    if not isinstance(raw_feature_scores, list):
        raise SystemExit(f"{context}.feature_scores must be a list")
    feature_score_features: set[str] = set()
    for item_index, raw_item in enumerate(raw_feature_scores):
        item_context = f"{context}.feature_scores[{item_index}]"
        if not isinstance(raw_item, dict):
            raise SystemExit(f"{item_context} must be an object")
        feature = _required_string(raw_item, "feature", item_context)
        if feature not in columns or feature in feature_score_features:
            raise SystemExit(f"{item_context}.feature is missing or duplicated")
        feature_score_features.add(feature)
        try:
            _finite_scalar(
                raw_item.get("state_information_ratio"), f"{item_context}.state_information_ratio"
            )
            _finite_scalar(raw_item.get("eta_squared"), f"{item_context}.eta_squared")
        except ValueError as error:
            raise SystemExit(str(error)) from error
        try:
            _aligned_probability_matrix(
                raw_item.get("teacher_timestamps"),
                raw_item.get("teacher_probabilities"),
                f"{item_context}.teacher",
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error

    raw_nmi = dossier.get("prefix_nmi", [])
    if not isinstance(raw_nmi, list):
        raise SystemExit(f"{context}.prefix_nmi must be a list")
    for item_index, raw_item in enumerate(raw_nmi):
        item_context = f"{context}.prefix_nmi[{item_index}]"
        if not isinstance(raw_item, dict):
            raise SystemExit(f"{item_context} must be an object")
        try:
            _aligned_probability_matrix(
                raw_item.get("candidate_timestamps"),
                raw_item.get("candidate_probabilities"),
                f"{item_context}.candidate",
            )
            _aligned_probability_matrix(
                raw_item.get("teacher_timestamps"),
                raw_item.get("teacher_probabilities"),
                f"{item_context}.teacher",
            )
            nmi = _finite_scalar(raw_item.get("soft_regime_nmi"), f"{item_context}.soft_regime_nmi")
        except ValueError as error:
            raise SystemExit(str(error)) from error
        if nmi < 0.0 or nmi > 1.0 + _AUDIT_TOLERANCE:
            raise SystemExit(f"{item_context}.soft_regime_nmi is outside [0,1]")

    if "likelihoods" in dossier and "gaussian_likelihoods" in dossier:
        raise SystemExit(f"{context} cannot contain both likelihoods and gaussian_likelihoods")
    raw_likelihoods = dossier.get("likelihoods", dossier.get("gaussian_likelihoods", []))
    if not isinstance(raw_likelihoods, list):
        raise SystemExit(f"{context}.likelihoods must be a list")
    for item_index, raw_item in enumerate(raw_likelihoods):
        if not isinstance(raw_item, dict):
            raise SystemExit(f"{context}.likelihoods[{item_index}] must be an object")
        _validate_likelihood_contract(raw_item, f"{context}.likelihoods[{item_index}]")
    return (
        features,
        timestamp_column,
        expected_distance,
        tuple(silhouettes),
        cast(list[dict[str, object]], raw_feature_scores),
        cast(list[dict[str, object]], raw_nmi),
        cast(list[dict[str, object]], raw_likelihoods),
    )


def _immutable_columns(columns: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not isinstance(columns, dict) or not columns:
        raise ValueError("snapshot columns must be a non-empty mapping")
    result: dict[str, np.ndarray] = {}
    row_count: int | None = None
    for name, raw_values in columns.items():
        if not isinstance(name, str) or not name:
            raise ValueError("snapshot column names must be non-empty strings")
        values = np.asarray(raw_values, dtype=object)
        if values.ndim != 1 or not len(values):
            raise ValueError("snapshot columns must be non-empty one-dimensional arrays")
        if row_count is None:
            row_count = len(values)
        elif len(values) != row_count:
            raise ValueError("snapshot columns must have one common row count")
        frozen = np.array(values, dtype=object, copy=True)
        frozen.setflags(write=False)
        result[name] = frozen
    return result


def _absolute_error(actual: object, expected: object, name: str) -> float:
    try:
        actual_value = _finite_scalar(actual, f"actual {name}")
        expected_value = _finite_scalar(expected, f"expected {name}")
    except ValueError as error:
        raise SystemExit(str(error)) from error
    error = abs(actual_value - expected_value)
    if not math.isfinite(error):
        raise SystemExit(f"{name} comparison produced a nonfinite error")
    return error


def _assemble_distances(
    distance_results: tuple[tuple[int, tuple[tuple[int, int, float], ...]], ...],
    dossier_features: tuple[tuple[str, ...], ...],
    expected_pairs: tuple[tuple[tuple[int, int], ...], ...],
) -> tuple[np.ndarray, ...]:
    distances = tuple(
        np.zeros((len(features), len(features)), dtype=float) for features in dossier_features
    )
    seen: list[set[tuple[int, int]]] = [set() for _ in dossier_features]
    for dossier_index, values in distance_results:
        if dossier_index < 0 or dossier_index >= len(distances):
            raise SystemExit("distance worker returned an invalid dossier index")
        for left_index, right_index, value in values:
            pair = (left_index, right_index)
            if pair not in set(expected_pairs[dossier_index]) or pair in seen[dossier_index]:
                raise SystemExit("distance worker returned an invalid or duplicate pair")
            distance = _finite_scalar(value, "computed distance")
            if not 0.0 <= distance <= 1.0:
                raise SystemExit("computed distance is outside [0,1]")
            distances[dossier_index][left_index, right_index] = distance
            distances[dossier_index][right_index, left_index] = distance
            seen[dossier_index].add(pair)
    for dossier_index, pairs in enumerate(expected_pairs):
        if seen[dossier_index] != set(pairs):
            raise SystemExit("distance worker did not return every expected pair")
    return distances


def verify_expectations(
    columns: dict[str, np.ndarray],
    expected: object,
    *,
    max_workers: int | None = None,
) -> dict[str, object]:
    """Verify all dossiers, using independent processes for CPU-heavy primitives."""

    immutable_columns = _immutable_columns(columns)
    dossiers = _audit_dossiers(expected)
    distance_tasks: list[tuple[int, tuple[str, ...], tuple[tuple[int, int], ...]]] = []
    feature_tasks: list[tuple[int, str, dict[str, object], str]] = []
    nmi_tasks: list[tuple[int, dict[str, object]]] = []
    likelihood_tasks: list[tuple[int, dict[str, object]]] = []
    dossier_features: list[tuple[str, ...]] = []
    expected_distances: list[np.ndarray] = []
    expected_pairs: list[tuple[tuple[int, int], ...]] = []
    dossier_silhouettes: list[tuple[tuple[int, tuple[int, ...], float], ...]] = []
    silhouette_specs: list[tuple[int, int, int, tuple[int, ...], float]] = []
    feature_task_indices: list[tuple[int, ...]] = []
    nmi_task_indices: list[tuple[int, ...]] = []
    likelihood_task_indices: list[tuple[int, ...]] = []
    for dossier_index, dossier in enumerate(dossiers):
        (
            features,
            timestamp_column,
            expected_distance,
            silhouettes,
            raw_feature_scores,
            raw_nmi,
            raw_likelihoods,
        ) = _validate_dossier_contract(dossier, immutable_columns, dossier_index)
        dossier_features.append(features)
        expected_distances.append(expected_distance)
        dossier_silhouettes.append(silhouettes)
        pairs = tuple(
            (left_index, right_index)
            for left_index in range(len(features))
            for right_index in range(left_index)
        )
        expected_pairs.append(pairs)
        # Use several chunks per worker so a large feature universe can keep
        # all available processes busy without creating one future per pair.
        target_chunks = max(1, 4 * _audit_requested_worker_limit(max_workers))
        chunk_size = max(1, (len(pairs) + target_chunks - 1) // target_chunks)
        distance_tasks.extend(
            (
                dossier_index,
                features,
                pairs[offset : offset + chunk_size],
            )
            for offset in range(0, len(pairs), chunk_size)
        )
        for cluster_index, (cluster_count, labels, mean) in enumerate(silhouettes):
            silhouette_specs.append((dossier_index, cluster_index, cluster_count, labels, mean))
        dossier_feature_indices: list[int] = []
        for raw_item in raw_feature_scores:
            if not isinstance(raw_item, dict):
                raise SystemExit("validated feature score is not an object")
            feature_value = raw_item.get("feature")
            if not isinstance(feature_value, str):
                raise SystemExit("validated feature score has no feature name")
            feature = feature_value
            dossier_feature_indices.append(len(feature_tasks))
            feature_tasks.append((len(feature_tasks), feature, raw_item, timestamp_column))
        feature_task_indices.append(tuple(dossier_feature_indices))
        dossier_nmi_indices: list[int] = []
        for raw_item in raw_nmi:
            if not isinstance(raw_item, dict):
                raise SystemExit("validated soft-NMI expectation is not an object")
            dossier_nmi_indices.append(len(nmi_tasks))
            nmi_tasks.append((len(nmi_tasks), raw_item))
        nmi_task_indices.append(tuple(dossier_nmi_indices))
        dossier_likelihood_indices: list[int] = []
        for raw_item in raw_likelihoods:
            if not isinstance(raw_item, dict):
                raise SystemExit("validated likelihood expectation is not an object")
            dossier_likelihood_indices.append(len(likelihood_tasks))
            likelihood_tasks.append((len(likelihood_tasks), raw_item))
        likelihood_task_indices.append(tuple(dossier_likelihood_indices))

    task_count = (
        len(distance_tasks)
        + len(feature_tasks)
        + len(nmi_tasks)
        + len(likelihood_tasks)
        + len(silhouette_specs)
    )
    worker_count = _audit_worker_count(max_workers, task_count)
    if worker_count > 1:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=_audit_process_context(),
            initializer=_initialize_audit_worker,
            initargs=(immutable_columns,),
        ) as executor:
            distance_futures = tuple(
                executor.submit(_process_distance_chunk, task) for task in distance_tasks
            )
            feature_futures = tuple(
                executor.submit(_process_feature_score, task) for task in feature_tasks
            )
            nmi_futures = tuple(executor.submit(_process_nmi, task) for task in nmi_tasks)
            likelihood_futures = tuple(
                executor.submit(_process_likelihood, task) for task in likelihood_tasks
            )
            distance_results = tuple(future.result() for future in distance_futures)
            feature_results = tuple(future.result() for future in feature_futures)
            nmi_results = tuple(future.result() for future in nmi_futures)
            likelihood_results = tuple(future.result() for future in likelihood_futures)
            distances = _assemble_distances(
                distance_results,
                tuple(dossier_features),
                tuple(expected_pairs),
            )
            silhouette_tasks = tuple(
                (
                    dossier_index,
                    cluster_index,
                    cluster_count,
                    tuple(tuple(float(value) for value in row) for row in distances[dossier_index]),
                    labels,
                )
                for dossier_index, cluster_index, cluster_count, labels, _mean in silhouette_specs
            )
            silhouette_futures = tuple(
                executor.submit(_process_silhouette, task) for task in silhouette_tasks
            )
            silhouette_results = tuple(future.result() for future in silhouette_futures)
    else:
        distance_results = tuple(
            (
                dossier_index,
                _distance_chunk(immutable_columns, features, pairs),
            )
            for dossier_index, features, pairs in distance_tasks
        )
        distances = _assemble_distances(
            distance_results,
            tuple(dossier_features),
            tuple(expected_pairs),
        )
        feature_results = tuple(
            (
                item_index,
                *independent_feature_score(
                    immutable_columns[feature],
                    immutable_columns[timestamp_column],
                    tuple(item["teacher_timestamps"]),
                    tuple(tuple(row) for row in item["teacher_probabilities"]),
                ),
            )
            for item_index, feature, item, timestamp_column in feature_tasks
        )
        nmi_results = tuple(
            (
                item_index,
                independent_soft_nmi(
                    tuple(item["candidate_timestamps"]),
                    tuple(tuple(row) for row in item["candidate_probabilities"]),
                    tuple(item["teacher_timestamps"]),
                    tuple(tuple(row) for row in item["teacher_probabilities"]),
                ),
            )
            for item_index, item in nmi_tasks
        )
        likelihood_results = tuple(
            (item_index, independent_hmm_log_likelihood(item))
            for item_index, item in likelihood_tasks
        )
        silhouette_results = tuple(
            (
                dossier_index,
                cluster_index,
                _silhouette_value(
                    distances[dossier_index],
                    np.asarray(labels, dtype=np.int64),
                    cluster_count,
                ),
            )
            for dossier_index, cluster_index, cluster_count, labels, _mean in silhouette_specs
        )

    distance_errors: list[float] = []
    silhouette_errors: list[float] = []
    feature_score_errors: list[float] = []
    nmi_errors: list[float] = []
    likelihood_errors: list[float] = []
    feature_values = {
        index: (information_ratio, eta_squared)
        for index, information_ratio, eta_squared in feature_results
    }
    nmi_values = {index: value for index, value in nmi_results}
    likelihood_values = {index: value for index, value in likelihood_results}
    silhouette_values = {
        (dossier_index, cluster_index): value
        for dossier_index, cluster_index, value in silhouette_results
    }
    if set(feature_values) != set(range(len(feature_tasks))):
        raise SystemExit("feature-score workers returned incomplete or duplicate results")
    if set(nmi_values) != set(range(len(nmi_tasks))):
        raise SystemExit("soft-NMI workers returned incomplete or duplicate results")
    if set(likelihood_values) != set(range(len(likelihood_tasks))):
        raise SystemExit("likelihood workers returned incomplete or duplicate results")
    expected_silhouette_keys = {
        (dossier_index, cluster_index)
        for dossier_index, cluster_index, _count, _labels, _mean in silhouette_specs
    }
    if set(silhouette_values) != expected_silhouette_keys:
        raise SystemExit("silhouette workers returned incomplete or duplicate results")
    for dossier_index, _dossier in enumerate(dossiers):
        features = dossier_features[dossier_index]
        distance = distances[dossier_index]
        distance_error = float(np.max(np.abs(distance - expected_distances[dossier_index])))
        if not math.isfinite(distance_error):
            raise SystemExit("distance comparison produced a nonfinite error")
        distance_errors.append(distance_error)
        for cluster_index, (_count, _labels, expected_mean) in enumerate(
            dossier_silhouettes[dossier_index]
        ):
            silhouette_errors.append(
                _absolute_error(
                    silhouette_values[(dossier_index, cluster_index)],
                    expected_mean,
                    "silhouette",
                )
            )

        for task_index in feature_task_indices[dossier_index]:
            item = feature_tasks[task_index][2]
            information_ratio, eta_squared = feature_values[task_index]
            feature_score_errors.extend(
                (
                    _absolute_error(
                        information_ratio,
                        item["state_information_ratio"],
                        "state-information ratio",
                    ),
                    _absolute_error(eta_squared, item["eta_squared"], "eta-squared"),
                )
            )
        for task_index in nmi_task_indices[dossier_index]:
            item = nmi_tasks[task_index][1]
            nmi_errors.append(
                _absolute_error(nmi_values[task_index], item["soft_regime_nmi"], "soft-NMI")
            )
        for task_index in likelihood_task_indices[dossier_index]:
            item = likelihood_tasks[task_index][1]
            likelihood_errors.append(
                _absolute_error(
                    likelihood_values[task_index], item["log_likelihood"], "HMM likelihood"
                )
            )
        if distance_errors[-1] > _AUDIT_TOLERANCE:
            raise SystemExit(f"distance audit failed: max_abs_error={distance_errors[-1]:.12g}")
    maximum_distance_error = max(distance_errors, default=0.0)
    maximum_silhouette_error = max(silhouette_errors, default=0.0)
    maximum_feature_score_error = max(feature_score_errors, default=0.0)
    maximum_nmi_error = max(nmi_errors, default=0.0)
    maximum_likelihood_error = max(likelihood_errors, default=0.0)
    if maximum_feature_score_error > 1.0e-10:
        raise SystemExit("feature-score audit failed")
    if maximum_silhouette_error > _AUDIT_TOLERANCE:
        raise SystemExit("silhouette audit failed")
    if maximum_nmi_error > 1.0e-10:
        raise SystemExit("soft-NMI audit failed")
    if maximum_likelihood_error > 1.0e-10:
        raise SystemExit("HMM likelihood audit failed")

    return {
        "audited_outer_fold_indices": [int(dossier["outer_fold_index"]) for dossier in dossiers]
        if isinstance(expected, dict) and "fold_audits" in expected
        else [],
        "audited_outer_fold_count": len(dossiers),
        "distance_max_abs_error": maximum_distance_error,
        "silhouette_max_abs_error": maximum_silhouette_error,
        "feature_score_max_abs_error": maximum_feature_score_error,
        "soft_nmi_max_abs_error": maximum_nmi_error,
        "gaussian_likelihood_max_abs_error": maximum_likelihood_error,
        "status": "verified",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--expectations",
        type=Path,
        required=True,
        help="JSON containing independently auditable expected primitive outputs",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="maximum independent audit processes (default: all allowed CPUs)",
    )
    args = parser.parse_args()
    columns = _read_snapshot(args.snapshot)
    expected = json.loads(args.expectations.read_text(encoding="utf-8"))
    print(json.dumps(verify_expectations(columns, expected, max_workers=args.workers)))


if __name__ == "__main__":
    main()
