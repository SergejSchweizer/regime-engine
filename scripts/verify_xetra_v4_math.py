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
    if requested is not None and requested < 1:
        raise ValueError("workers must be positive")
    available = _audit_available_cpu_count()
    return min(requested or available, available, task_count)


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


def _distance_chunk(
    columns: dict[str, np.ndarray],
    features: tuple[str, ...],
    pairs: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int, float], ...]:
    values: list[tuple[int, int, float]] = []
    for left_index, right_index in pairs:
        left, right = _complete(columns[features[left_index]], columns[features[right_index]])
        left_rank = rankdata(left, method="average")
        right_rank = rankdata(right, method="average")
        correlation = float(np.corrcoef(left_rank, right_rank)[0, 1])
        values.append((left_index, right_index, 1.0 - abs(correlation)))
    return tuple(values)


def _process_distance_chunk(
    task: tuple[int, tuple[str, ...], tuple[tuple[int, int], ...]],
) -> tuple[int, tuple[tuple[int, int, float], ...]]:
    if _AUDIT_COLUMNS is None:
        raise RuntimeError("audit process worker was not initialized")
    dossier_index, features, pairs = task
    return dossier_index, _distance_chunk(_AUDIT_COLUMNS, features, pairs)


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
    state_count = len(starts)
    if values.ndim != 2 or starts.ndim != 1:
        raise ValueError("GMM likelihood primitive dimensions are invalid")
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
    state_count = len(starts)
    if values.ndim != 2 or starts.ndim != 1:
        raise ValueError("Student-t likelihood primitive dimensions are invalid")
    if transitions.shape != (state_count, state_count):
        raise ValueError("Student-t transition dimensions are invalid")
    if state_means.shape != (state_count, values.shape[1]):
        raise ValueError("Student-t mean dimensions are invalid")
    if state_covariances.shape != (state_count, values.shape[1], values.shape[1]):
        raise ValueError("Student-t covariance dimensions are invalid")
    if degrees.shape != (state_count,) or np.any(degrees <= 2.0):
        raise ValueError("Student-t degrees of freedom are invalid")
    emissions = np.empty((len(values), state_count), dtype=float)
    dimension = values.shape[1]
    for state, (mean, covariance, degree) in enumerate(
        zip(state_means, state_covariances, degrees, strict=True)
    ):
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
    if log_emissions.ndim != 2 or log_emissions.shape[1] != len(starts):
        raise ValueError("forward emission dimensions are invalid")
    if log_emissions.shape[0] == 0:
        raise ValueError("forward emissions cannot be empty")
    if starts.ndim != 1 or np.any(starts < 0.0) or not np.isclose(starts.sum(), 1.0):
        raise ValueError("forward start probabilities are invalid")
    if transitions.shape != (len(starts), len(starts)) or np.any(transitions < 0.0):
        raise ValueError("forward transition probabilities are invalid")
    if not np.allclose(transitions.sum(axis=1), 1.0):
        raise ValueError("forward transition rows must be normalized")
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
    return total


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
    """Return the fold dossiers, validating the deterministic audit contract."""

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
    try:
        indices = tuple(int(item["outer_fold_index"]) for item in dossiers)
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit("each fold audit must declare outer_fold_index") from error
    if indices != tuple(sorted(set(indices))):
        raise SystemExit("fold audit outer-fold indices must be unique and sorted")
    declared = expected.get("audit_outer_fold_indices")
    if declared != list(indices):
        raise SystemExit("audit_outer_fold_indices does not match fold_audits")
    return dossiers


def verify_expectations(
    columns: dict[str, np.ndarray],
    expected: object,
    *,
    max_workers: int | None = None,
) -> dict[str, object]:
    """Verify all dossiers, using independent processes for CPU-heavy primitives."""

    dossiers = _audit_dossiers(expected)
    distance_tasks: list[tuple[int, tuple[str, ...], tuple[tuple[int, int], ...]]] = []
    feature_tasks: list[tuple[int, str, dict[str, object], str]] = []
    nmi_tasks: list[tuple[int, dict[str, object]]] = []
    likelihood_tasks: list[tuple[int, dict[str, object]]] = []
    dossier_features: list[tuple[str, ...]] = []
    for dossier_index, dossier in enumerate(dossiers):
        features = tuple(str(feature) for feature in dossier["feature_order"])
        dossier_features.append(features)
        pairs = tuple(
            (left_index, right_index)
            for left_index in range(len(features))
            for right_index in range(left_index)
        )
        # Use several chunks per worker so a large feature universe can keep
        # all available processes busy without creating one future per pair.
        chunk_size = max(1, (len(pairs) + 31) // 32)
        distance_tasks.extend(
            (
                dossier_index,
                features,
                pairs[offset : offset + chunk_size],
            )
            for offset in range(0, len(pairs), chunk_size)
        )

        raw_feature_scores = dossier.get("feature_scores", [])
        if not isinstance(raw_feature_scores, list):
            raise SystemExit("feature score expectations must be a list")
        timestamp_column = str(dossier["timestamp_column"])
        for raw_item in raw_feature_scores:
            if not isinstance(raw_item, dict):
                raise SystemExit("feature score expectation must be an object")
            feature_tasks.append(
                (len(feature_tasks), str(raw_item["feature"]), raw_item, timestamp_column)
            )

        raw_nmi = dossier.get("prefix_nmi", [])
        if not isinstance(raw_nmi, list):
            raise SystemExit("soft-NMI expectations must be a list")
        for raw_item in raw_nmi:
            if not isinstance(raw_item, dict):
                raise SystemExit("soft-NMI expectation must be an object")
            nmi_tasks.append((len(nmi_tasks), raw_item))

        likelihood_items = dossier.get("likelihoods", dossier.get("gaussian_likelihoods", []))
        if not isinstance(likelihood_items, list):
            raise SystemExit("likelihood expectations must be a list")
        for raw_item in likelihood_items:
            if not isinstance(raw_item, dict):
                raise SystemExit("likelihood expectation must be an object")
            likelihood_tasks.append((len(likelihood_tasks), raw_item))

    task_count = len(distance_tasks) + len(feature_tasks) + len(nmi_tasks) + len(likelihood_tasks)
    worker_count = _audit_worker_count(max_workers, task_count)
    if worker_count > 1:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=_audit_process_context(),
            initializer=_initialize_audit_worker,
            initargs=(columns,),
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
    else:
        distance_results = tuple(
            (
                dossier_index,
                _distance_chunk(columns, features, pairs),
            )
            for dossier_index, features, pairs in distance_tasks
        )
        feature_results = tuple(
            (
                item_index,
                *independent_feature_score(
                    columns[feature],
                    columns[timestamp_column],
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

    distances = {
        index: np.zeros((len(features), len(features)), dtype=float)
        for index, features in enumerate(dossier_features)
    }
    for dossier_index, values in distance_results:
        for left_index, right_index, value in values:
            distances[dossier_index][left_index, right_index] = value
            distances[dossier_index][right_index, left_index] = value

    distance_errors: list[float] = []
    feature_score_errors: list[float] = []
    nmi_errors: list[float] = []
    likelihood_errors: list[float] = []
    feature_values = {
        index: (information_ratio, eta_squared)
        for index, information_ratio, eta_squared in feature_results
    }
    nmi_values = {index: value for index, value in nmi_results}
    likelihood_values = {index: value for index, value in likelihood_results}
    for dossier_index, dossier in enumerate(dossiers):
        features = dossier_features[dossier_index]
        distance = distances[dossier_index]
        expected_distance = np.asarray(dossier["distance"], dtype=float)
        if expected_distance.shape != distance.shape:
            raise SystemExit("distance expectation dimensions are invalid")
        distance_errors.append(float(np.max(np.abs(distance - expected_distance))))

        for cluster in dossier.get("silhouette_clusters", []):
            labels = np.asarray(cluster["labels"], dtype=int)
            if labels.shape != (len(features),):
                raise SystemExit("silhouette label dimensions are invalid")
            samples = silhouette_samples(distance, labels, metric="precomputed")
            actual = float(np.mean(np.where(labels == -1, 0.0, samples)))
            if abs(actual - float(cluster["mean"])) > 1.0e-10:
                raise SystemExit("silhouette audit failed")

        timestamp_column = str(dossier["timestamp_column"])
        raw_feature_scores = dossier.get("feature_scores", [])
        assert isinstance(raw_feature_scores, list)
        feature_offset = sum(
            len(cast(list[object], other.get("feature_scores", [])))
            for other in dossiers[:dossier_index]
        )
        for item_index, item in enumerate(raw_feature_scores):
            assert isinstance(item, dict)
            information_ratio, eta_squared = feature_values[feature_offset + item_index]
            feature_score_errors.extend(
                (
                    abs(information_ratio - float(item["state_information_ratio"])),
                    abs(eta_squared - float(item["eta_squared"])),
                )
            )
        raw_nmi = dossier.get("prefix_nmi", [])
        assert isinstance(raw_nmi, list)
        nmi_offset = sum(
            len(cast(list[object], other.get("prefix_nmi", [])))
            for other in dossiers[:dossier_index]
        )
        for item_index, item in enumerate(raw_nmi):
            assert isinstance(item, dict)
            actual = nmi_values[nmi_offset + item_index]
            nmi_errors.append(abs(actual - float(item["soft_regime_nmi"])))
        likelihood_items = dossier.get("likelihoods", dossier.get("gaussian_likelihoods", []))
        assert isinstance(likelihood_items, list)
        likelihood_offset = sum(
            len(cast(list[object], other.get("likelihoods", other.get("gaussian_likelihoods", []))))
            for other in dossiers[:dossier_index]
        )
        for item_index, item in enumerate(likelihood_items):
            assert isinstance(item, dict)
            actual = likelihood_values[likelihood_offset + item_index]
            likelihood_errors.append(abs(actual - float(item["log_likelihood"])))
        if distance_errors[-1] > 1.0e-10:
            raise SystemExit(f"distance audit failed: max_abs_error={distance_errors[-1]:.12g}")
    maximum_distance_error = max(distance_errors, default=0.0)
    maximum_feature_score_error = max(feature_score_errors, default=0.0)
    maximum_nmi_error = max(nmi_errors, default=0.0)
    maximum_likelihood_error = max(likelihood_errors, default=0.0)
    if maximum_feature_score_error > 1.0e-10:
        raise SystemExit("feature-score audit failed")
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
