from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.evaluation.report.contracts import thaw_json
from market_regime_engine.evaluation.report.feature_selection import build_feature_selection_report
from market_regime_engine.feature_selection.contracts import (
    FeatureBlock,
    FeatureSelectionPolicy,
)
from market_regime_engine.feature_selection.freeze import freeze_first_train_features
from market_regime_engine.feature_selection.stability import (
    FeatureSelectionStabilityDiagnostics,
    ThresholdSensitivityDiagnostic,
)

_SHA = "a" * 64


def _singleton_fixture() -> tuple[pd.DataFrame, FeatureSelectionPolicy]:
    rng = np.random.default_rng(17)
    rows = 600
    shared = rng.normal(size=rows)
    values: dict[str, np.ndarray] = {
        "f0": shared,
        "f1": shared + rng.normal(scale=0.001, size=rows),
        "f2": shared + rng.normal(scale=0.002, size=rows),
    }
    for index in range(3, 8):
        values[f"f{index}"] = rng.normal(size=rows)
    frame = pd.DataFrame(values)
    policy = FeatureSelectionPolicy(
        policy_id="xetra_semantic_medoid_v1",
        blocks=tuple(FeatureBlock(f"block_{index}", (f"f{index}",)) for index in range(8)),
    )
    return frame, policy


def _multifeature_fixture() -> tuple[pd.DataFrame, FeatureSelectionPolicy]:
    rng = np.random.default_rng(23)
    rows = 600
    base = rng.normal(size=rows)
    frame = pd.DataFrame(
        {
            "a0": base,
            "a1": base + rng.normal(scale=0.03, size=rows),
            "a_bad": rng.normal(size=rows),
            **{f"f{index}": rng.normal(size=rows) for index in range(1, 8)},
        }
    )
    frame.loc[:99, "a_bad"] = np.nan
    policy = FeatureSelectionPolicy(
        policy_id="xetra_semantic_medoid_v1",
        blocks=(
            FeatureBlock("block_0", ("a0", "a1", "a_bad")),
            *(FeatureBlock(f"block_{index}", (f"f{index}",)) for index in range(1, 8)),
        ),
    )
    return frame, policy


def _freeze(frame: pd.DataFrame, policy: FeatureSelectionPolicy):
    return freeze_first_train_features(
        frame,
        policy,
        source_build_id="build-1",
        data_sha256=_SHA,
        evaluation_plan_hash=_SHA,
    )


def test_report_reconstructs_stage1_and_full_stage2_processing_trace() -> None:
    frame, policy = _singleton_fixture()
    frozen = _freeze(frame, policy)
    report = build_feature_selection_report(frame, policy, frozen)
    payload = thaw_json(report.evidence)

    assert payload["policy"]["maximum_cross_block_abs_spearman"] == 0.85
    assert payload["stage1_blocks"][0]["singleton_matrix"] is True
    assert payload["stage1_blocks"][0]["absolute_spearman_matrix"] == [[1.0]]
    assert payload["stage1_blocks"][0]["distance_matrix_one_minus_abs_spearman"] == [[0.0]]
    assert payload["stage1_blocks"][0]["stage1_decision"]["winner"] == "f0"

    trace = payload["stage2"]["pair_processing_trace"]
    assert len(trace) >= 3
    assert any(item["processed"] for item in trace)
    assert any(not item["processed"] for item in trace)
    assert all(item["abs_spearman"] > 0.85 for item in trace)
    assert payload["stage2"]["final_features"] == list(frozen.final_features)
    removed = [
        item for item in payload["stage2"]["final_dispositions"] if item["disposition"] == "removed"
    ]
    assert removed
    assert all(item["removal_decision_index"] is not None for item in removed)


def test_report_explains_multifeature_block_eligibility_matrices_and_anchors() -> None:
    frame, policy = _multifeature_fixture()
    frozen = _freeze(frame, policy)
    report = build_feature_selection_report(frame, policy, frozen)
    block = thaw_json(report.evidence)["stage1_blocks"][0]

    assert block["singleton_matrix"] is False
    assert block["eligible_feature_order"] == ["a0", "a1"]
    assert len(block["absolute_spearman_matrix"]) == 2
    scores = {item["feature_name"]: item for item in block["feature_scores"]}
    assert scores["a_bad"]["eligible"] is False
    assert scores["a_bad"]["exclusion_reason"] == "coverage_below_minimum"
    assert scores["a_bad"]["nonnull_count"] == 500
    assert scores["a0"]["source_row_denominator"] == 600
    decision = block["stage1_decision"]
    assert decision["minimum_medoid_score_anchor"] >= 0.0
    assert decision["configured_position_tiebreak"] == 0
    assert decision["winner"] == "a0"


def test_diagnostics_are_labelled_nondecision_and_hash_bound() -> None:
    frame, policy = _singleton_fixture()
    frozen = _freeze(frame, policy)
    threshold = tuple(
        ThresholdSensitivityDiagnostic(
            threshold=value,
            selected_features=frozen.final_features,
            conflicts=(),
            failure_reason=None,
            canonical=value == 0.85,
        )
        for value in (0.80, 0.85, 0.90)
    )
    diagnostics = FeatureSelectionStabilityDiagnostics(
        frozen_final_features=frozen.final_features,
        frozen_definition_hash=frozen.feature_selection_definition_hash,
        frozen_execution_hash=frozen.feature_selection_execution_hash,
        threshold_sensitivity=threshold,
        shadow_folds=(),
    )
    report = build_feature_selection_report(frame, policy, frozen, diagnostics)
    diagnostic_payload = thaw_json(report.evidence)["diagnostics"]
    assert diagnostic_payload["diagnostic_only"] is True
    assert [item["threshold"] for item in diagnostic_payload["threshold_sensitivity"]] == [
        0.8,
        0.85,
        0.9,
    ]

    mismatched = replace(diagnostics, frozen_execution_hash="b" * 64)
    with pytest.raises(ValueError, match="different selection hashes"):
        build_feature_selection_report(frame, policy, frozen, mismatched)


def test_report_fails_closed_when_inputs_do_not_reproduce_frozen_evidence() -> None:
    frame, policy = _singleton_fixture()
    frozen = _freeze(frame, policy)

    with pytest.raises(ValueError, match="row count"):
        build_feature_selection_report(frame.iloc[:-1], policy, frozen)

    wrong_policy = replace(policy, policy_id="xetra_semantic_medoid_v2")
    with pytest.raises(ValueError, match="identity mismatch"):
        build_feature_selection_report(frame, wrong_policy, frozen)

    wrong_hash = replace(frozen, feature_selection_definition_hash="b" * 64)
    with pytest.raises(ValueError, match="definition hash"):
        build_feature_selection_report(frame, policy, wrong_hash)

    changed = frame.copy()
    changed.loc[0, "f0"] = np.nan
    with pytest.raises(ValueError, match="coverage evidence mismatch|complete-case count mismatch"):
        build_feature_selection_report(changed, policy, frozen)


def test_report_contains_no_raw_feature_vectors() -> None:
    frame, policy = _singleton_fixture()
    frozen = _freeze(frame, policy)
    payload = thaw_json(build_feature_selection_report(frame, policy, frozen).evidence)
    text = repr(payload)
    assert "raw_source_rows" not in text
    assert "raw_feature_rows" not in text
    assert len(payload["stage1_blocks"][0]["feature_scores"]) == 1
