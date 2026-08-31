from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from market_regime_engine.evaluation.walk_forward import WalkForwardEvaluation
from market_regime_engine.mlflow_support.plots import (
    render_em_convergence,
    render_em_convergence_comparison,
    summarize_em_convergence,
)


def _fold(
    fold_id: str,
    history: tuple[float, ...] | None,
    *,
    train_count: int = 10,
    seed: int = 11,
    valid: bool = True,
):
    if not valid:
        return SimpleNamespace(
            fold_id=fold_id,
            valid=False,
            train_model_observation_count=train_count,
            multistart_result=None,
        )
    if history is None:
        multistart = SimpleNamespace(
            winner=SimpleNamespace(seed=seed, iterations=0, em_log_likelihood_history=())
        )
    else:
        multistart = SimpleNamespace(
            winner=SimpleNamespace(
                seed=seed,
                iterations=len(history),
                em_log_likelihood_history=history,
            )
        )
    return SimpleNamespace(
        fold_id=fold_id,
        valid=True,
        train_model_observation_count=train_count,
        multistart_result=multistart,
    )


def _evaluation(candidate_id: str, folds: tuple[object, ...]) -> WalkForwardEvaluation:
    return cast(
        WalkForwardEvaluation,
        SimpleNamespace(candidate_id=candidate_id, feature_order=("vix_delta_1obs",), folds=folds),
    )


def test_summary_normalizes_per_fold_and_never_fills_missing_tails() -> None:
    evaluation = _evaluation(
        "gaussian_hmm_k2_full",
        (
            _fold("fold_001", (-100.0, -90.0, -80.0), train_count=10, seed=11),
            _fold("fold_002", (-200.0, -180.0), train_count=20, seed=23),
            _fold("fold_003", None, train_count=10, seed=37),
            _fold("fold_004", None, valid=False),
        ),
    )
    summary = summarize_em_convergence(evaluation)
    assert summary.available
    assert summary.iterations == (1, 2, 3)
    assert summary.fold_series[0].normalized_log_likelihood == (-10.0, -9.0, -8.0)
    assert summary.fold_series[1].normalized_log_likelihood == (-10.0, -9.0)
    assert summary.median == (-10.0, -9.0, -8.0)
    assert summary.q25 == (-10.0, -9.0, -8.0)
    assert summary.q75 == (-10.0, -9.0, -8.0)
    assert summary.missing_trace_fold_ids == ("fold_003",)
    assert summary.invalid_fold_count == 1


def test_unavailable_candidate_is_explicit_and_renders_no_fake_series(tmp_path: Path) -> None:
    evaluation = _evaluation(
        "gaussian_hmm_k3_full",
        (_fold("fold_001", None), _fold("fold_002", None, valid=False)),
    )
    entry, summary = render_em_convergence(evaluation, "vix_delta_1obs", tmp_path)
    assert not summary.available
    assert summary.median == ()
    assert entry.available_candidate_ids == ()
    assert entry.unavailable_candidate_ids == ("gaussian_hmm_k3_full",)
    assert Path(entry.png_path).is_file()
    assert Path(entry.svg_path).is_file()
    assert entry.diagnostic_label == "optimization diagnostic only — not model selection"


def test_candidate_plot_records_png_svg_and_deterministic_source_hash(tmp_path: Path) -> None:
    evaluation = _evaluation(
        "gmm_hmm_k3_m2_full",
        (_fold("fold_001", (-30.0, -20.0), train_count=10),),
    )
    first, first_summary = render_em_convergence(evaluation, "vix_delta_1obs", tmp_path / "a")
    second, second_summary = render_em_convergence(evaluation, "vix_delta_1obs", tmp_path / "b")
    assert first_summary == second_summary
    assert first.source_artifact_hash == second.source_artifact_hash
    assert Path(first.png_path).is_file()
    assert Path(first.svg_path).is_file()
    assert first.x_axis_label == "EM iteration"
    assert first.y_axis_label == "TRAIN log likelihood per observation"


def test_all_model_comparison_preserves_supplied_canonical_order_and_unavailable_ids(
    tmp_path: Path,
) -> None:
    evaluations = (
        _evaluation(
            "gaussian_hmm_k2_full",
            (_fold("fold_001", (-30.0, -20.0), train_count=10),),
        ),
        _evaluation("gaussian_hmm_k3_full", (_fold("fold_001", None),)),
        _evaluation(
            "gmm_hmm_k2_m2_full",
            (_fold("fold_001", (-60.0, -40.0), train_count=20),),
        ),
    )
    entry, summaries = render_em_convergence_comparison(evaluations, "vix_delta_1obs", tmp_path)
    assert entry.candidate_ids == (
        "gaussian_hmm_k2_full",
        "gaussian_hmm_k3_full",
        "gmm_hmm_k2_m2_full",
    )
    assert tuple(summary.candidate_id for summary in summaries) == entry.candidate_ids
    assert entry.unavailable_candidate_ids == ("gaussian_hmm_k3_full",)
    assert Path(entry.png_path).is_file()
    assert Path(entry.svg_path).is_file()


def test_feature_mismatch_duplicate_candidate_and_bad_history_fail_closed(tmp_path: Path) -> None:
    wrong_feature = cast(
        WalkForwardEvaluation,
        SimpleNamespace(
            candidate_id="gaussian_hmm_k2_full",
            feature_order=("other",),
            folds=(_fold("fold_001", (-1.0,)),),
        ),
    )
    with pytest.raises(ValueError, match="feature"):
        render_em_convergence(wrong_feature, "vix_delta_1obs", tmp_path)

    duplicate = _evaluation("gaussian_hmm_k2_full", (_fold("fold_001", (-1.0,), train_count=1),))
    with pytest.raises(ValueError, match="unique"):
        render_em_convergence_comparison((duplicate, duplicate), "vix_delta_1obs", tmp_path)

    bad = _evaluation(
        "gaussian_hmm_k2_full",
        (
            SimpleNamespace(
                fold_id="fold_001",
                valid=True,
                train_model_observation_count=1,
                multistart_result=SimpleNamespace(
                    winner=SimpleNamespace(seed=11, iterations=2, em_log_likelihood_history=(-1.0,))
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="length"):
        summarize_em_convergence(bad)
