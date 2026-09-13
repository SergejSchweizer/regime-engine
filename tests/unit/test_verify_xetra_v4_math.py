from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from math import lgamma
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.metrics import silhouette_samples

from market_regime_engine.inference.filtering import causal_filter
from market_regime_engine.models.artifacts import GaussianHMMArtifact


def _module() -> Any:
    path = Path(__file__).parents[2] / "scripts" / "verify_xetra_v4_math.py"
    spec = importlib.util.spec_from_file_location("verify_xetra_v4_math", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load independent math verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
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


def test_parallel_audit_verification_uses_canonical_results() -> None:
    module = _module()
    columns = {
        "timestamp_m1": np.asarray([0, 1, 2, 3], dtype=object),
        "feature_a": np.asarray([0.0, 1.0, 2.0, 3.0]),
        "feature_b": np.asarray([3.0, 2.0, 1.0, 0.0]),
        "feature_c": np.asarray([0.0, 0.5, 1.5, 3.0]),
    }
    features = ("feature_a", "feature_b", "feature_c")
    distance = module.independent_distance(columns, features).tolist()
    dossiers = [
        {
            "outer_fold_index": index,
            "timestamp_column": "timestamp_m1",
            "feature_order": list(features),
            "distance": distance,
            "silhouette_clusters": [
                {
                    "cluster_count": 2,
                    "labels": [0, 0, 1],
                    "mean": float(
                        np.mean(
                            silhouette_samples(
                                np.asarray(distance, dtype=float),
                                np.asarray([0, 0, 1], dtype=int),
                                metric="precomputed",
                            )
                        )
                    ),
                }
            ],
        }
        for index in range(3)
    ]

    report = module.verify_expectations(
        columns,
        {"fold_audits": dossiers, "audit_outer_fold_indices": [0, 1, 2]},
        max_workers=2,
    )

    assert report["status"] == "verified"
    assert report["audited_outer_fold_indices"] == [0, 1, 2]
    assert report["distance_max_abs_error"] == 0.0
    assert report["silhouette_max_abs_error"] == 0.0


def test_audit_rejects_nonfinite_expected_distance_instead_of_passing_nan() -> None:
    module = _module()
    columns = {
        "timestamp_m1": np.asarray([0, 1, 2], dtype=object),
        "feature_a": np.asarray([0.0, 1.0, 2.0]),
        "feature_b": np.asarray([2.0, 1.0, 0.0]),
    }
    with pytest.raises(SystemExit, match="finite"):
        module.verify_expectations(
            columns,
            {
                "outer_fold_index": 0,
                "timestamp_column": "timestamp_m1",
                "feature_order": ["feature_a", "feature_b"],
                "distance": [[0.0, float("nan")], [float("nan"), 0.0]],
            },
            max_workers=1,
        )


def test_audit_rejects_rank_pairs_without_two_complete_observations() -> None:
    module = _module()
    columns = {
        "timestamp_m1": np.asarray([0, 1], dtype=object),
        "feature_a": np.asarray([1.0, None], dtype=object),
        "feature_b": np.asarray([2.0, None], dtype=object),
    }
    with pytest.raises(ValueError, match="at least two complete"):
        module.verify_expectations(
            columns,
            {
                "outer_fold_index": 0,
                "timestamp_column": "timestamp_m1",
                "feature_order": ["feature_a", "feature_b"],
                "distance": [[0.0, 0.0], [0.0, 0.0]],
            },
            max_workers=1,
        )


def test_soft_nmi_rejects_empty_timestamp_intersection() -> None:
    module = _module()
    with pytest.raises(ValueError, match="shared timestamps"):
        module.independent_soft_nmi(
            ("2026-01-01",),
            ((1.0, 0.0),),
            ("2026-01-02",),
            ((1.0, 0.0),),
        )


def test_likelihood_rejects_non_normalized_transition_rows() -> None:
    module = _module()
    with pytest.raises(ValueError, match="transition rows"):
        module.independent_gaussian_log_likelihood(
            np.asarray([[0.0]], dtype=float),
            np.asarray([1.0]),
            np.asarray([[0.5]]),
            np.asarray([[0.0]]),
            np.asarray([[[1.0]]]),
        )


def test_audit_worker_count_respects_affinity_cgroup_and_task_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setattr(module, "_audit_affinity_cpu_count", lambda: 86)
    monkeypatch.setattr(module, "_audit_cgroup_cpu_limit", lambda: 4)

    assert module._audit_available_cpu_count() == 4
    assert module._audit_worker_count(None, 100) == 4
    assert module._audit_worker_count(86, 100) == 4
    assert module._audit_worker_count(2, 100) == 2
    assert module._audit_worker_count(None, 3) == 3


def test_audit_cgroup_reader_supports_v2_quota(tmp_path: Path) -> None:
    module = _module()
    cgroup_root = tmp_path / "cgroup"
    cgroup_root.mkdir()
    (cgroup_root / "cpu.max").write_text("350000 100000\n", encoding="utf-8")
    membership = tmp_path / "membership"
    membership.write_text("0::/worker\n", encoding="utf-8")
    nested = cgroup_root / "worker"
    nested.mkdir()
    (nested / "cpu.max").write_text("250000 100000\n", encoding="utf-8")

    assert module._audit_cgroup_cpu_limit(cgroup_root, membership) == 2


def test_audit_cgroup_reader_supports_v1_quota(tmp_path: Path) -> None:
    module = _module()
    cgroup_root = tmp_path / "cgroup"
    cpu_root = cgroup_root / "cpu" / "worker"
    cpu_root.mkdir(parents=True)
    (cpu_root / "cpu.cfs_quota_us").write_text("450000\n", encoding="utf-8")
    (cpu_root / "cpu.cfs_period_us").write_text("100000\n", encoding="utf-8")
    membership = tmp_path / "membership"
    membership.write_text("2:cpu,cpuacct:/worker\n", encoding="utf-8")

    assert module._audit_cgroup_cpu_limit(cgroup_root, membership) == 4


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
