from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.feature_selection import (
    FeatureBlock,
    FeatureScore,
    FeatureSelectionPolicy,
)
from market_regime_engine.feature_selection.selector import (
    _anchored_medoid_winner,
    average_rank_spearman,
    select_stage1,
    select_stage1_block,
)


def policy(blocks: tuple[FeatureBlock, ...] | None = None) -> FeatureSelectionPolicy:
    if blocks is None:
        blocks = tuple(FeatureBlock(f"b{index}", (f"f{index}",)) for index in range(8))
    return FeatureSelectionPolicy("xetra_semantic_medoid_v1", blocks)


def padded_policy(block: FeatureBlock) -> FeatureSelectionPolicy:
    pads = tuple(FeatureBlock(f"pad{i}", (f"p{i}",)) for i in range(7))
    return policy((block, *pads))


def test_average_rank_spearman_handles_ties_exactly() -> None:
    frame = pd.DataFrame(
        {
            "a": [1.0, 1.0, 3.0, 4.0],
            "b": [10.0, 10.0, 30.0, 40.0],
            "c": [4.0, 3.0, 2.0, 1.0],
        }
    )
    result = average_rank_spearman(frame)
    assert result.shape == (3, 3)
    assert result[0, 1] == pytest.approx(1.0)
    assert result[0, 2] < -0.94
    assert np.diag(result) == pytest.approx(np.ones(3))


def test_medoid_selects_most_representative_feature() -> None:
    rows = 600
    base = np.arange(rows, dtype=np.float64)
    frame = pd.DataFrame(
        {
            "a": base,
            "b": base + np.sin(base) * 0.01,
            "c": -base,
        }
    )
    block = FeatureBlock("semantic", ("a", "b", "c"))
    evidence = select_stage1_block(frame, block, padded_policy(block))
    assert evidence.complete_observation_count == rows
    assert evidence.winner == "a"
    scores = {score.feature_name: score for score in evidence.scores}
    assert all(score.eligible for score in scores.values())
    assert scores["a"].medoid_score is not None


def test_coverage_nonfinite_and_variance_exclusions_are_explicit() -> None:
    rows = 600
    frame = pd.DataFrame(
        {
            "good": np.arange(rows, dtype=np.float64),
            "low_coverage": [float(index) if index < 500 else np.nan for index in range(rows)],
            "nonfinite": np.arange(rows, dtype=np.float64),
            "constant": np.ones(rows),
        }
    )
    frame.loc[10, "nonfinite"] = np.inf
    block = FeatureBlock("semantic", ("good", "low_coverage", "nonfinite", "constant"))
    evidence = select_stage1_block(frame, block, padded_policy(block))
    scores = {score.feature_name: score for score in evidence.scores}
    assert scores["good"].eligible is True
    assert scores["low_coverage"].exclusion_reason == "coverage_below_minimum"
    assert scores["nonfinite"].exclusion_reason == "nonfinite_nonnull_value"
    assert scores["constant"].exclusion_reason == "variance_below_or_equal_minimum"
    assert evidence.winner == "good"


def test_singleton_eligible_feature_has_zero_medoid_score() -> None:
    frame = pd.DataFrame({"good": np.arange(600, dtype=float), "bad": np.ones(600)})
    block = FeatureBlock("semantic", ("good", "bad"))
    evidence = select_stage1_block(frame, block, padded_policy(block))
    good = next(score for score in evidence.scores if score.feature_name == "good")
    assert good.medoid_score == 0.0
    assert evidence.winner == "good"


def test_tie_break_uses_coverage_then_configured_position() -> None:
    rows = 600
    base = np.arange(rows, dtype=float)
    frame = pd.DataFrame({"a": base, "b": base})
    frame.loc[0:29, "a"] = np.nan
    block = FeatureBlock("semantic", ("a", "b"))
    evidence = select_stage1_block(frame, block, padded_policy(block))
    assert evidence.winner == "b"

    equal = pd.DataFrame({"a": base, "b": base})
    equal_evidence = select_stage1_block(equal, block, padded_policy(block))
    assert equal_evidence.winner == "a"


def test_anchored_medoid_tie_resolution_rejects_pairwise_tolerance_chain() -> None:
    scores = (
        FeatureScore("a", 0, 0.90, 1.0, 0.0, True),
        FeatureScore("b", 1, 0.95, 1.0, 0.75e-12, True),
        FeatureScore("c", 2, 0.99, 1.0, 1.5e-12, True),
    )

    winner = _anchored_medoid_winner(scores, tolerance=1e-12)
    reversed_winner = _anchored_medoid_winner(tuple(reversed(scores)), tolerance=1e-12)

    assert winner.feature_name == "b"
    assert reversed_winner.feature_name == "b"


def test_missing_columns_empty_rows_no_eligible_and_complete_row_gate_fail_closed() -> None:
    block = FeatureBlock("semantic", ("a", "b"))
    stage_policy = padded_policy(block)
    with pytest.raises(ValueError, match="non-empty"):
        select_stage1_block(pd.DataFrame(columns=["a", "b"]), block, stage_policy)
    with pytest.raises(ValueError, match="missing configured feature columns: b"):
        select_stage1_block(pd.DataFrame({"a": np.arange(600)}), block, stage_policy)
    with pytest.raises(ValueError, match="no eligible"):
        constant = pd.DataFrame({"a": np.ones(600), "b": np.ones(600)})
        select_stage1_block(constant, block, stage_policy)

    sparse = pd.DataFrame(
        {
            "a": np.arange(600, dtype=float),
            "b": np.arange(600, dtype=float),
        }
    )
    sparse.loc[:50, "a"] = np.nan
    sparse.loc[51:100, "b"] = np.nan
    with pytest.raises(ValueError, match="complete observations"):
        select_stage1_block(sparse, block, stage_policy)


def test_undefined_spearman_fails_without_fallback() -> None:
    frame = pd.DataFrame(
        {
            "a": np.tile([0.0, 1.0], 300),
            "b": np.tile([0.0, 1.0], 300),
        }
    )
    matrix = average_rank_spearman(frame)
    assert matrix[0, 1] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="finite"):
        average_rank_spearman(pd.DataFrame({"a": [1.0, 1.0], "b": [2.0, 2.0]}))
    with pytest.raises(ValueError, match="two rows and two columns"):
        average_rank_spearman(pd.DataFrame({"a": [1.0, 2.0]}))
    with pytest.raises(ValueError, match="finite"):
        average_rank_spearman(pd.DataFrame({"a": [1.0, np.inf], "b": [1.0, 2.0]}))


def test_select_stage1_preserves_eight_block_order() -> None:
    blocks = tuple(FeatureBlock(f"b{index}", (f"f{index}",)) for index in range(8))
    frame = pd.DataFrame({f"f{index}": np.arange(600, dtype=float) + index for index in range(8)})
    result = select_stage1(frame, policy(blocks))
    assert tuple(item.block_id for item in result) == tuple(f"b{index}" for index in range(8))
    assert tuple(item.winner for item in result) == tuple(f"f{index}" for index in range(8))
