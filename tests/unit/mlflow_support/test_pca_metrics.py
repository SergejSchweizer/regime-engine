from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from market_regime_engine.mlflow_support.model_metrics import model_metric_points
from market_regime_engine.mlflow_support.pca_metrics import (
    PCAMetricComparison,
    pca_comparison_metric_points,
    pca_metric_points,
)
from market_regime_engine.mlflow_support.pca_plots import (
    render_pca_diagnostics,
    render_pca_oos_comparison,
)
from tests.unit.evaluation.test_walk_forward import evaluate as evaluate_raw
from tests.unit.evaluation.test_walk_forward import source_rows as raw_source_rows
from tests.unit.evaluation.test_walk_forward_pca import evaluate as evaluate_pca
from tests.unit.evaluation.test_walk_forward_pca import source_rows as pca_source_rows


def comparison() -> PCAMetricComparison:
    raw = evaluate_raw(raw_source_rows(1323))
    pca = evaluate_pca(pca_source_rows(1323))
    return PCAMetricComparison(
        raw_only=raw,
        raw_plus_pca=replace(pca, source_build_id=raw.source_build_id),
    )


def test_pca_metric_projection_contains_loadings_and_explained_variance() -> None:
    pca = comparison().raw_plus_pca
    points = pca_metric_points(pca)
    keys = {point.key for point in points}

    assert "pca_retained_component_count" in keys
    assert "pca_cumulative_explained_variance" in keys
    assert "pca_explained_variance_ratio_component_001" in keys
    assert "pca_loading_component_001_feature_001" in keys
    assert any(point.key.startswith("pca_loading_") for point in model_metric_points(pca))


def test_matched_pca_oos_projection_is_fold_aligned() -> None:
    points = pca_comparison_metric_points(comparison())
    by_key = {
        point.key: tuple(item for item in points if item.key == point.key) for point in points
    }

    assert len(by_key["pca_comparison_delta_oos_predictive_loglik_per_obs"]) == 1
    assert len(by_key["pca_comparison_delta_mean"]) == 1
    assert len(by_key["pca_comparison_win_count"]) == 1


def test_pca_plot_renderers_write_nonempty_pngs(tmp_path: Path) -> None:
    matched = comparison()
    entries = render_pca_diagnostics(matched.raw_plus_pca, tmp_path)
    entries += (render_pca_oos_comparison(matched, tmp_path),)

    assert len(entries) == 3
    assert all(Path(entry.png_path).is_file() for entry in entries)
    assert all(Path(entry.png_path).stat().st_size > 0 for entry in entries)


def test_pca_comparison_rejects_different_source_plans() -> None:
    matched = comparison()
    changed = replace(matched.raw_plus_pca, evaluation_plan_hash="c" * 64)
    with pytest.raises(ValueError, match="identical evaluation plan"):
        PCAMetricComparison(raw_only=matched.raw_only, raw_plus_pca=changed)
