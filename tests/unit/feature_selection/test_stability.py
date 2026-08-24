from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.feature_selection.contracts import FeatureBlock, FeatureSelectionPolicy
from market_regime_engine.feature_selection.freeze import freeze_first_train_features
from market_regime_engine.feature_selection.stability import (
    CANONICAL_STAGE2_THRESHOLD,
    THRESHOLD_SENSITIVITY_LEVELS,
    build_stability_diagnostics,
    jaccard_overlap,
    later_fold_shadow_diagnostics,
    threshold_sensitivity,
)


def policy() -> FeatureSelectionPolicy:
    return FeatureSelectionPolicy(
        "xetra_semantic_medoid_v1",
        tuple(FeatureBlock(f"b{index}", (f"f{index}",)) for index in range(8)),
    )


def frame(rows: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(20260824)
    return pd.DataFrame({f"f{index}": rng.normal(size=rows) for index in range(8)})


def frozen_result():
    return freeze_first_train_features(
        frame(),
        policy(),
        source_build_id="build-1",
        data_sha256="a" * 64,
        evaluation_plan_hash="b" * 64,
    )


def test_threshold_sensitivity_uses_exact_levels_and_only_085_is_canonical() -> None:
    diagnostics = threshold_sensitivity(frozen_result(), policy())
    assert tuple(item.threshold for item in diagnostics) == THRESHOLD_SENSITIVITY_LEVELS
    assert tuple(item.canonical for item in diagnostics) == (False, True, False)
    assert next(item for item in diagnostics if item.canonical).threshold == CANONICAL_STAGE2_THRESHOLD
    assert all(item.selected_features is not None for item in diagnostics)


def test_shadow_rerun_records_ordered_selection_and_exact_jaccard() -> None:
    frozen = frozen_result()
    diagnostics = later_fold_shadow_diagnostics(
        (("fold-002", frame()),),
        frozen,
        policy(),
    )
    assert len(diagnostics) == 1
    result = diagnostics[0]
    assert result.fold_id == "fold-002"
    assert result.selected_features == frozen.final_features
    assert result.jaccard_overlap == 1.0
    assert result.failure_reason is None


def test_shadow_failure_is_diagnostic_only() -> None:
    frozen = frozen_result()
    bad = frame().drop(columns=["f7"])
    diagnostics = later_fold_shadow_diagnostics(
        (("fold-003", bad),),
        frozen,
        policy(),
    )
    result = diagnostics[0]
    assert result.selected_features is None
    assert result.jaccard_overlap is None
    assert result.failure_reason is not None
    assert "missing configured feature columns" in result.failure_reason
    assert frozen.final_features == tuple(f"f{index}" for index in range(8))


def test_build_stability_diagnostics_preserves_frozen_identity() -> None:
    frozen = frozen_result()
    result = build_stability_diagnostics(
        frozen,
        policy(),
        (("fold-002", frame()),),
    )
    assert result.frozen_final_features == frozen.final_features
    assert result.frozen_definition_hash == frozen.feature_selection_definition_hash
    assert result.frozen_execution_hash == frozen.feature_selection_execution_hash
    assert tuple(item.threshold for item in result.threshold_sensitivity) == (0.80, 0.85, 0.90)
    assert result.shadow_folds[0].jaccard_overlap == 1.0


def test_jaccard_overlap_uses_set_intersection_over_union() -> None:
    assert jaccard_overlap(("a", "b"), ("b", "c")) == pytest.approx(1.0 / 3.0)
    with pytest.raises(ValueError, match="at least one feature"):
        jaccard_overlap((), ())
