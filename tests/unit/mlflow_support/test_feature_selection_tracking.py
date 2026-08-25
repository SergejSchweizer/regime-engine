from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from market_regime_engine.feature_selection.contracts import (
    BlockSelectionEvidence,
    FeatureScore,
    FeatureSelectionEvidence,
    FeatureSelectionResult,
)
from market_regime_engine.feature_selection.stability import (
    FeatureSelectionStabilityDiagnostics,
    ThresholdSensitivityDiagnostic,
)
from market_regime_engine.mlflow_support.feature_selection_tracking import (
    track_feature_selection_evidence,
)


class FakeTrackingPort:
    def __init__(self) -> None:
        self.artifacts: list[tuple[str, str, str]] = []

    def start_run(self, *, run_name: str, parent_run_id: str | None = None) -> str:
        del run_name, parent_run_id
        return "unused"

    def log_params(self, run_id: str, params: dict[str, str]) -> None:
        del run_id, params

    def log_metric_points(self, run_id: str, points: object) -> None:
        del run_id, points

    def log_artifact(self, run_id: str, local_path: str, artifact_path: str) -> None:
        self.artifacts.append((run_id, local_path, artifact_path))


def selection_result() -> FeatureSelectionResult:
    blocks = tuple(
        BlockSelectionEvidence(
            block_id=f"block_{index}",
            complete_observation_count=1260,
            scores=(
                FeatureScore(
                    feature_name=f"feature_{index}",
                    configured_position=0,
                    coverage=1.0,
                    population_variance=1.0,
                    medoid_score=0.0,
                    eligible=True,
                ),
            ),
            winner=f"feature_{index}",
        )
        for index in range(8)
    )
    medoids = tuple(block.winner for block in blocks)
    identity = tuple(
        tuple(1.0 if row == column else 0.1 for column in range(8)) for row in range(8)
    )
    evidence = FeatureSelectionEvidence(
        first_train_source_row_count=1260,
        block_evidence=blocks,
        preliminary_medoids=medoids,
        stage2_complete_observation_count=1260,
        stage2_abs_spearman_matrix=identity,
        conflicts=(),
        final_features=medoids,
    )
    return FeatureSelectionResult(
        policy_id="xetra_semantic_medoid_v1",
        final_features=medoids,
        feature_selection_definition_hash="a" * 64,
        feature_selection_execution_hash="b" * 64,
        evidence=evidence,
    )


def stability(result: FeatureSelectionResult) -> FeatureSelectionStabilityDiagnostics:
    return FeatureSelectionStabilityDiagnostics(
        frozen_final_features=result.final_features,
        frozen_definition_hash=result.feature_selection_definition_hash,
        frozen_execution_hash=result.feature_selection_execution_hash,
        threshold_sensitivity=tuple(
            ThresholdSensitivityDiagnostic(
                threshold=threshold,
                selected_features=result.final_features,
                conflicts=(),
                failure_reason=None,
                canonical=threshold == 0.85,
            )
            for threshold in (0.80, 0.85, 0.90)
        ),
        shadow_folds=(),
    )


def matrices(result: FeatureSelectionResult) -> dict[str, tuple[tuple[float, ...], ...]]:
    return {block.block_id: ((1.0,),) for block in result.evidence.block_evidence}


def test_complete_feature_selection_audit_is_deterministic_and_logged(tmp_path: Path) -> None:
    result = selection_result()
    port = FakeTrackingPort()
    tracked = track_feature_selection_evidence(
        port,
        parent_run_id="parent-1",
        result=result,
        diagnostics=stability(result),
        within_block_abs_spearman=matrices(result),
        artifact_root=tmp_path,
    )

    assert len(tracked.artifacts) == 10
    assert Path(tracked.summary_path).is_file()
    assert Path(tracked.evidence_path).is_file()
    assert Path(tracked.diagnostics_path).is_file()
    assert Path(tracked.manifest_path).is_file()
    assert "48 source features -> 8 Stage-1 medoids" in Path(tracked.summary_path).read_text()
    for artifact in tracked.artifacts:
        assert artifact.png_path is not None and Path(artifact.png_path).is_file()
        assert artifact.title
        assert artifact.x_axis_label
        assert artifact.y_axis_label
        assert len(artifact.source_sha256) == 64
    manifest = json.loads(Path(tracked.manifest_path).read_text())
    assert len(manifest) == 10
    within_block_count = sum(
        item["artifact_type"] == "stage1_within_block_abs_spearman" for item in manifest
    )
    assert within_block_count == 8
    assert any(item["artifact_type"] == "stage2_cross_block_abs_spearman" for item in manifest)
    assert len(port.artifacts) == 14
    assert all(item[0] == "parent-1" for item in port.artifacts)


def test_diagnostic_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    result = selection_result()
    invalid = replace(stability(result), frozen_definition_hash="c" * 64)
    with pytest.raises(ValueError, match="definition hash"):
        track_feature_selection_evidence(
            FakeTrackingPort(),
            parent_run_id="parent-1",
            result=result,
            diagnostics=invalid,
            within_block_abs_spearman=matrices(result),
            artifact_root=tmp_path,
        )


def test_within_block_order_and_shape_are_enforced(tmp_path: Path) -> None:
    result = selection_result()
    reversed_matrices = dict(reversed(tuple(matrices(result).items())))
    with pytest.raises(ValueError, match="canonical block order"):
        track_feature_selection_evidence(
            FakeTrackingPort(),
            parent_run_id="parent-1",
            result=result,
            diagnostics=stability(result),
            within_block_abs_spearman=reversed_matrices,
            artifact_root=tmp_path,
        )
