from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from hashlib import sha256
from math import tanh

import pytest

from market_regime_engine.evaluations.process_parallel import cpu_process_pool
from market_regime_engine.feature_discovery.feature_subset_score import (
    FeatureSubsetCandidate,
    FeatureSubsetFoldEvidence,
    score_feature_subset,
)
from market_regime_engine.feature_discovery.sffs import (
    FeatureSubsetScore,
    SFFSResult,
    select_sffs,
)


def _fold(
    fold_id: str,
    *,
    latest: bool = False,
    valid: bool = True,
    target: float = -1.0,
    baseline: float = -1.5,
) -> FeatureSubsetFoldEvidence:
    if not valid:
        return FeatureSubsetFoldEvidence(
            fold_id, False, latest=latest, invalid_reason="invalid fit"
        )
    return FeatureSubsetFoldEvidence(
        fold_id,
        True,
        latest=latest,
        target_log_score=target,
        baseline_target_log_score=baseline,
        calibration_error=0.1,
        stability_score=0.8,
        support_score=0.9,
    )


def _candidate(*folds: FeatureSubsetFoldEvidence) -> FeatureSubsetCandidate:
    return FeatureSubsetCandidate(("a", "b"), "a" * 64, "build", "b" * 64, tuple(folds))


def _score_for_workers(features: tuple[str, ...]) -> FeatureSubsetScore:
    values = {
        ("a",): 1.0,
        ("b",): 3.0,
        ("c",): 2.0,
        ("b", "a"): 4.0,
        ("b", "c"): 5.0,
        ("b", "c", "a"): 6.0,
    }
    value = values.get(features, 0.0)
    return FeatureSubsetScore(
        features,
        value,
        forecast_score=value / 6.0,
        worst_fold_forecast_score=value / 6.0,
        calibration_score=value / 6.0,
        stability_score=value / 6.0,
        robustness_score=value / 6.0,
    )


def _result_hash(result: SFFSResult) -> str:
    payload = json.dumps(asdict(result), sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode()).hexdigest()


def _native_thread_snapshot() -> tuple[str | None, tuple[int, ...]]:
    try:
        from threadpoolctl import threadpool_info  # type: ignore[import-untyped]
    except ImportError:
        return os.environ.get("OMP_NUM_THREADS"), ()
    return (
        os.environ.get("OMP_NUM_THREADS"),
        tuple(int(item["num_threads"]) for item in threadpool_info() if "num_threads" in item),
    )


def test_score_is_recomputed_component_by_component_without_complexity_penalty() -> None:
    result = score_feature_subset(
        _candidate(_fold("m1"), _fold("m2"), _fold("m3"), _fold("m4", latest=True)),
        latest_fold_id="m4",
    )
    forecast = 0.5 + 0.5 * tanh(0.5)
    robustness = 0.50 * 1.0 + 0.25 * forecast + 0.25 * 0.9
    assert result.forecast_score == pytest.approx(forecast)
    assert result.calibration_score == pytest.approx(0.9)
    assert result.stability_score == pytest.approx(0.8)
    assert result.robustness_score == pytest.approx(robustness)
    assert result.total_score == pytest.approx(
        0.50 * forecast + 0.20 * 0.9 + 0.20 * 0.8 + 0.10 * robustness
    )


def test_score_eligibility_boundary_and_latest_fold_gate() -> None:
    boundary = score_feature_subset(
        _candidate(
            _fold("m1"),
            _fold("m2"),
            _fold("m3"),
            _fold("m4", latest=True, valid=False),
            _fold("m5"),
        ),
        latest_fold_id="m4",
    )
    assert boundary.eligible is False
    assert boundary.valid_fold_rate == pytest.approx(0.8)
    assert "latest inner fold is invalid" in boundary.rejection_reasons


def test_sffs_covers_no_winner_singleton_forward_backward_and_cap_paths() -> None:
    with pytest.raises(ValueError, match="eligible singleton"):
        select_sffs(("a", "b"), lambda _features: None, max_workers=1)

    singleton = select_sffs(
        tuple(f"f{index}" for index in range(12)),
        lambda features: FeatureSubsetScore(features, 1.0 / len(features)),
        max_features=10,
        max_workers=1,
    )
    assert len(singleton.selected_features) == 1

    floating = select_sffs(
        ("a", "b", "c"),
        lambda features: FeatureSubsetScore(
            features,
            {
                ("a",): 1.0,
                ("b",): 3.0,
                ("c",): 1.0,
                ("b", "a"): 4.0,
                ("b", "a", "c"): 5.0,
                ("a", "c"): 6.0,
            }.get(features, 0.0),
        ),
        max_features=3,
        max_workers=1,
    )
    assert floating.selected_features == ("a", "c")
    assert any(step.action == "add" for step in floating.steps)
    assert any(step.action == "remove" for step in floating.steps)
    assert all(len(step.selected_features) <= 10 for step in floating.steps)


def test_dimension_independent_score_cannot_be_overridden_by_raw_pll() -> None:
    with pytest.raises(ValueError, match="dimension-dependent"):
        FeatureSubsetScore(("raw_pll_winner",), 10_000.0, metric="raw_pll")

    result = select_sffs(
        ("lower_pll", "better_score"),
        lambda features: FeatureSubsetScore(
            features, 0.9 if features == ("better_score",) else 0.8
        ),
        max_workers=1,
    )
    assert result.selected_features == ("better_score",)


@pytest.mark.parametrize("worker_count", [1, 8, 32, 64, None])
def test_worker_counts_have_identical_paths_and_hashes(worker_count: int | None) -> None:
    result = select_sffs(("a", "b", "c"), _score_for_workers, max_workers=worker_count)
    reference = select_sffs(("a", "b", "c"), _score_for_workers, max_workers=1)
    assert result == reference
    assert _result_hash(result) == _result_hash(reference)


def _delayed_score_for_workers(features: tuple[str, ...]) -> FeatureSubsetScore:
    time.sleep((4 - len(features)) * 0.003)
    return _score_for_workers(features)


def test_randomized_completion_delays_preserve_sffs_identity() -> None:
    delayed = select_sffs(("a", "b", "c"), _delayed_score_for_workers, max_workers=8)
    reference = select_sffs(("a", "b", "c"), _score_for_workers, max_workers=1)
    assert delayed == reference
    assert _result_hash(delayed) == _result_hash(reference)


def test_process_worker_limits_native_numerical_threads_to_one() -> None:
    with cpu_process_pool(1) as executor:
        omp_threads, native_threads = executor.submit(_native_thread_snapshot).result()
    assert omp_threads == "1"
    assert all(thread_count == 1 for thread_count in native_threads)


def test_outer_test_mutation_cannot_change_inner_sffs_selection() -> None:
    outer_test = {"score": 100.0}

    def inner_score(features: tuple[str, ...]) -> FeatureSubsetScore:
        return FeatureSubsetScore(features, 2.0 if features == ("a",) else 1.0)

    before = select_sffs(("a", "b"), inner_score, max_workers=1)
    outer_test["score"] = -100.0
    after = select_sffs(("a", "b"), inner_score, max_workers=1)
    assert before == after
