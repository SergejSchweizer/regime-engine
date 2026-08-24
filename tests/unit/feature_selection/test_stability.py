from __future__ import annotations

from typing import Never

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.feature_selection import stability
from market_regime_engine.feature_selection.contracts import (
    BlockSelectionEvidence,
    FeatureBlock,
    FeatureScore,
    FeatureSelectionPolicy,
    FeatureSelectionResult,
)
from market_regime_engine.feature_selection.freeze import freeze_first_train_features
from market_regime_engine.feature_selection.stability import (
    CANONICAL_STAGE2_THRESHOLD,
    THRESHOLD_SENSITIVITY_LEVELS,
    FeatureSelectionStabilityDiagnostics,
    ShadowFoldDiagnostic,
    ThresholdSensitivityDiagnostic,
    _choose_removal,
    _diagnostic_prune,
    _winner_evidence,
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


def frozen_result() -> FeatureSelectionResult:
    return freeze_first_train_features(
        frame(),
        policy(),
        source_build_id="build-1",
        data_sha256="a" * 64,
        evaluation_plan_hash="b" * 64,
    )


def block_evidence(
    *,
    scores: tuple[float, ...] = (0.0,) * 8,
    coverages: tuple[float, ...] = (1.0,) * 8,
) -> tuple[BlockSelectionEvidence, ...]:
    return tuple(
        BlockSelectionEvidence(
            block_id=f"b{index}",
            complete_observation_count=600,
            scores=(
                FeatureScore(
                    feature_name=f"f{index}",
                    configured_position=0,
                    coverage=coverages[index],
                    population_variance=1.0,
                    medoid_score=scores[index],
                    eligible=True,
                ),
            ),
            winner=f"f{index}",
        )
        for index in range(8)
    )


def identity_matrix() -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(1.0 if row == column else 0.1 for column in range(8)) for row in range(8))


def test_threshold_sensitivity_uses_exact_levels_and_only_085_is_canonical() -> None:
    diagnostics = threshold_sensitivity(frozen_result(), policy())
    assert tuple(item.threshold for item in diagnostics) == THRESHOLD_SENSITIVITY_LEVELS
    assert tuple(item.canonical for item in diagnostics) == (False, True, False)
    canonical = next(item for item in diagnostics if item.canonical)
    assert canonical.threshold == CANONICAL_STAGE2_THRESHOLD
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
    thresholds = tuple(item.threshold for item in result.threshold_sensitivity)
    assert thresholds == (0.80, 0.85, 0.90)
    assert result.shadow_folds[0].jaccard_overlap == 1.0


def test_jaccard_overlap_uses_set_intersection_over_union() -> None:
    assert jaccard_overlap(("a", "b"), ("b", "c")) == pytest.approx(1.0 / 3.0)
    with pytest.raises(ValueError, match="at least one feature"):
        jaccard_overlap((), ())


def test_diagnostic_contracts_fail_closed() -> None:
    valid = ThresholdSensitivityDiagnostic(0.85, ("f0",), (), None, True)
    assert valid.canonical is True
    with pytest.raises(ValueError, match="exactly"):
        ThresholdSensitivityDiagnostic(0.86, ("f0",), (), None, False)
    with pytest.raises(ValueError, match="only threshold"):
        ThresholdSensitivityDiagnostic(0.80, ("f0",), (), None, True)
    with pytest.raises(ValueError, match="either selected"):
        ThresholdSensitivityDiagnostic(0.80, None, (), None, False)
    with pytest.raises(ValueError, match="either selected"):
        ThresholdSensitivityDiagnostic(0.80, ("f0",), (), "failure", False)
    with pytest.raises(ValueError, match="zero features"):
        ThresholdSensitivityDiagnostic(0.80, (), (), None, False)

    assert ShadowFoldDiagnostic("fold-2", ("f0",), 1.0, None).jaccard_overlap == 1.0
    with pytest.raises(ValueError, match="fold_id"):
        ShadowFoldDiagnostic(" fold", ("f0",), 1.0, None)
    with pytest.raises(ValueError, match="zero features"):
        ShadowFoldDiagnostic("fold", (), 1.0, None)
    with pytest.raises(ValueError, match="finite Jaccard"):
        ShadowFoldDiagnostic("fold", ("f0",), None, None)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        ShadowFoldDiagnostic("fold", ("f0",), 1.1, None)
    with pytest.raises(ValueError, match="cannot carry"):
        ShadowFoldDiagnostic("fold", ("f0",), 1.0, "failure")
    with pytest.raises(ValueError, match="requires only"):
        ShadowFoldDiagnostic("fold", None, 0.5, "failure")
    with pytest.raises(ValueError, match="requires only"):
        ShadowFoldDiagnostic("fold", None, None, None)


def test_stability_summary_validates_threshold_order_and_hash_identity() -> None:
    sensitivity = tuple(
        ThresholdSensitivityDiagnostic(value, ("f0",), (), None, value == 0.85)
        for value in THRESHOLD_SENSITIVITY_LEVELS
    )
    valid = FeatureSelectionStabilityDiagnostics(
        ("f0",),
        "a" * 64,
        "b" * 64,
        sensitivity,
        (),
    )
    assert valid.frozen_definition_hash == "a" * 64
    with pytest.raises(ValueError, match="cannot be empty"):
        FeatureSelectionStabilityDiagnostics((), "a" * 64, "b" * 64, sensitivity, ())
    with pytest.raises(ValueError, match="preserve"):
        FeatureSelectionStabilityDiagnostics(
            ("f0",),
            "a" * 64,
            "b" * 64,
            tuple(reversed(sensitivity)),
            (),
        )
    with pytest.raises(ValueError, match="SHA-256"):
        FeatureSelectionStabilityDiagnostics(("f0",), "A" * 64, "b" * 64, sensitivity, ())


def test_diagnostic_prune_validation_and_all_tie_break_paths() -> None:
    blocks = block_evidence()
    medoids = tuple(f"f{index}" for index in range(8))
    with pytest.raises(ValueError, match="one of"):
        _diagnostic_prune(blocks, medoids, identity_matrix(), threshold=0.86, tolerance=1e-12)
    with pytest.raises(ValueError, match="canonical order"):
        _diagnostic_prune(
            blocks,
            tuple(reversed(medoids)),
            identity_matrix(),
            threshold=0.85,
            tolerance=1e-12,
        )
    with pytest.raises(ValueError, match="finite 8x8"):
        _diagnostic_prune(blocks, medoids, ((1.0,),), threshold=0.85, tolerance=1e-12)

    winners = _winner_evidence(blocks)
    assert _choose_removal("f0", "f1", winners, 1e-12) == (
        "f1",
        "later canonical block",
    )
    score_winners = _winner_evidence(block_evidence(scores=(0.4, 0.2, *([0.0] * 6))))
    assert _choose_removal("f0", "f1", score_winners, 1e-12)[0] == "f0"
    assert _choose_removal("f1", "f0", score_winners, 1e-12)[0] == "f0"
    coverage_winners = _winner_evidence(block_evidence(coverages=(0.9, 1.0, *([1.0] * 6))))
    assert _choose_removal("f0", "f1", coverage_winners, 1e-12)[0] == "f0"
    assert _choose_removal("f1", "f0", coverage_winners, 1e-12)[0] == "f0"
    with pytest.raises(ValueError, match="exactly eight"):
        _winner_evidence(blocks[:-1])


def test_threshold_diagnostic_failure_is_recorded_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> Never:
        raise ValueError("diagnostic-only failure")

    monkeypatch.setattr(stability, "_diagnostic_prune", fail)
    diagnostics = threshold_sensitivity(frozen_result(), policy())
    assert len(diagnostics) == 3
    assert all(item.selected_features is None for item in diagnostics)
    assert all(item.failure_reason == "ValueError: diagnostic-only failure" for item in diagnostics)
