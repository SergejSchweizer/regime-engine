from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from market_regime_engine.evaluation.report.contracts import (
    EVALUATION_REPORT_SCHEMA_VERSION,
    CandidateReport,
    CrossCandidateComparisonReport,
    EvaluationReport,
    FeatureSelectionReport,
    FoldReport,
    ReportIntegrity,
    ReportMetadata,
    canonical_json_bytes,
    object_payload,
    seal_report,
    verify_report,
)

_SHA = "a" * 64
_CANDIDATES = (
    "gaussian_hmm_k2_full",
    "gaussian_hmm_k3_full",
    "gaussian_hmm_k4_full",
)


def _report(evidence: dict[str, object] | None = None) -> EvaluationReport:
    metadata = ReportMetadata(
        parent_run_id="run-1",
        parent_run_status="FINISHED",
        profile_id="xetra",
        profile_config_version=1,
        source_dataset="regime-gold",
        source_build_id="build-1",
        data_sha256=_SHA,
        source_schema_version=1,
        source_feature_version=1,
        source_synced_at_utc=datetime(2026, 8, 27, 12, tzinfo=UTC),
        data_time_semantics="current_vintage_observation_day",
        repository_git_sha="deadbeef",
        build_provenance="unit-test",
        evaluation_plan_hash=_SHA,
        evaluation_cutoff=datetime(2026, 8, 27, 13, tzinfo=UTC),
        feature_order=("vix_level",),
        feature_selection_definition_hash=_SHA,
        feature_selection_execution_hash=_SHA,
    )
    feature = FeatureSelectionReport(
        policy_id="xetra_semantic_medoid_v1",
        final_features=("vix_level",),
        feature_selection_definition_hash=_SHA,
        feature_selection_execution_hash=_SHA,
        evidence=object_payload(evidence or {"a": 1, "b": 2}),
    )
    folds = tuple(
        FoldReport("fold_001", 1, True, None, object_payload({"candidate": candidate_id}))
        for candidate_id in _CANDIDATES
    )
    candidates = tuple(
        CandidateReport(
            candidate_id=candidate_id,
            state_count=index + 2,
            source_build_id="build-1",
            evaluation_plan_hash=_SHA,
            feature_order=("vix_level",),
            feature_selection_definition_hash=_SHA,
            feature_selection_execution_hash=_SHA,
            folds=(folds[index],),
            summary=object_payload({"valid_fold_count": 1}),
        )
        for index, candidate_id in enumerate(_CANDIDATES)
    )
    comparison = CrossCandidateComparisonReport(
        champion_candidate_id=_CANDIDATES[0],
        ranked_candidate_ids=_CANDIDATES,
        common_valid_fold_ids=("fold_001",),
        evidence=object_payload({"diagnostic_only": False}),
    )
    return EvaluationReport(
        schema_version=EVALUATION_REPORT_SCHEMA_VERSION,
        metadata=metadata,
        configured_candidate_ids=_CANDIDATES,
        planned_fold_ids=("fold_001",),
        feature_selection=feature,
        candidates=candidates,
        comparison=comparison,
    )


def test_canonical_bytes_ignore_mapping_insertion_order() -> None:
    first = canonical_json_bytes(seal_report(_report({"b": 2, "a": 1})))
    second = canonical_json_bytes(seal_report(_report({"a": 1, "b": 2})))
    assert first == second
    assert json.loads(first)["schema_version"] == EVALUATION_REPORT_SCHEMA_VERSION
    assert json.loads(first)["metadata"]["source_synced_at_utc"].endswith("Z")


def test_nonfinite_evidence_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        object_payload({"metric": float("nan")})


def test_duplicate_or_missing_fold_identity_fails_closed() -> None:
    report = _report()
    with pytest.raises(ValueError, match="duplicate-free"):
        replace(report, planned_fold_ids=("fold_001", "fold_001"))
    with pytest.raises(ValueError, match="exact planned fold"):
        replace(report, planned_fold_ids=("fold_001", "fold_002"))


def test_integrity_hash_detects_tampering() -> None:
    sealed = seal_report(_report())
    digest = verify_report(sealed)
    assert len(digest) == 64
    corrupted = replace(sealed, integrity=ReportIntegrity("0" * 64))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_report(corrupted)


def test_binary_and_secret_fields_are_forbidden() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        object_payload({"password": "not-allowed"})
    with pytest.raises(ValueError, match="binary"):
        object_payload({"artifact": b"bytes"})
