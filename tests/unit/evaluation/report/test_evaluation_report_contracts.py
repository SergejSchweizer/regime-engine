from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone

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
    canonical_payload_bytes,
    freeze_json,
    object_payload,
    report_payload_dict,
    seal_report,
    thaw_json,
    verify_report,
)

_SHA = "a" * 64
_CANDIDATES = (
    "gaussian_hmm_k2_full",
    "gaussian_hmm_k3_full",
    "gaussian_hmm_k4_full",
)


def _metadata() -> ReportMetadata:
    return ReportMetadata(
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


def _report(evidence: dict[str, object] | None = None) -> EvaluationReport:
    metadata = _metadata()
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
    decoded = json.loads(first)
    assert decoded["schema_version"] == EVALUATION_REPORT_SCHEMA_VERSION
    assert decoded["metadata"]["source_synced_at_utc"].endswith("Z")
    assert "integrity" not in json.loads(canonical_payload_bytes(seal_report(_report())))
    assert report_payload_dict(seal_report(_report()), include_integrity=True)["integrity"]


def test_frozen_json_round_trip_and_input_rejections() -> None:
    frozen = object_payload(
        {
            "timestamp": datetime(2026, 8, 27, tzinfo=UTC),
            "nested": {"values": [1, 2.5, True, None]},
        }
    )
    thawed = thaw_json(frozen)
    assert thawed["timestamp"] == "2026-08-27T00:00:00Z"
    assert thawed["nested"]["values"] == [1, 2.5, True, None]

    with pytest.raises(ValueError, match="finite"):
        object_payload({"metric": float("nan")})
    with pytest.raises(ValueError, match="forbidden"):
        object_payload({"password": "not-allowed"})
    with pytest.raises(ValueError, match="binary"):
        object_payload({"artifact": b"bytes"})
    with pytest.raises(TypeError, match="keys must be strings"):
        freeze_json({1: "value"})
    with pytest.raises(TypeError, match="unsupported"):
        freeze_json({"not-json"})
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        freeze_json(datetime(2026, 8, 27))


def test_metadata_validates_text_hash_versions_time_and_features() -> None:
    metadata = _metadata()
    with pytest.raises(ValueError, match="non-empty trimmed"):
        replace(metadata, parent_run_id=" run")
    with pytest.raises(ValueError, match="positive"):
        replace(metadata, profile_config_version=0)
    with pytest.raises(ValueError, match="positive"):
        replace(metadata, source_schema_version=0)
    with pytest.raises(ValueError, match="SHA-256"):
        replace(metadata, data_sha256="bad")
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        replace(metadata, evaluation_cutoff=datetime(2026, 8, 27))
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        replace(
            metadata,
            source_synced_at_utc=datetime(
                2026, 8, 27, tzinfo=timezone(timedelta(hours=2))
            ),
        )
    with pytest.raises(ValueError, match="duplicate-free"):
        replace(metadata, feature_order=("vix_level", "vix_level"))


def test_section_contracts_fail_closed_on_invalid_identity() -> None:
    report = _report()
    feature = report.feature_selection
    with pytest.raises(ValueError, match="non-empty"):
        replace(feature, policy_id="")
    with pytest.raises(ValueError, match="SHA-256"):
        replace(feature, feature_selection_definition_hash="bad")

    valid_fold = report.candidates[0].folds[0]
    with pytest.raises(ValueError, match="positive"):
        replace(valid_fold, fold_index=0)
    with pytest.raises(ValueError, match="invalid fold requires"):
        replace(valid_fold, valid=False, failure_reason=None)
    invalid_fold = FoldReport(
        "fold_001", 1, False, "fit_failed", object_payload({"available": False})
    )
    with pytest.raises(ValueError, match="valid fold has no"):
        replace(invalid_fold, valid=True)

    candidate = report.candidates[0]
    with pytest.raises(ValueError, match="K=2,3,4,5"):
        replace(candidate, state_count=6)
    with pytest.raises(ValueError, match="duplicate-free"):
        replace(candidate, folds=(valid_fold, valid_fold))
    with pytest.raises(ValueError, match="SHA-256"):
        replace(candidate, evaluation_plan_hash="bad")

    comparison = report.comparison
    with pytest.raises(ValueError, match="champion"):
        replace(comparison, champion_candidate_id=_CANDIDATES[1])
    with pytest.raises(ValueError, match="duplicate-free"):
        replace(comparison, ranked_candidate_ids=(_CANDIDATES[0], _CANDIDATES[0]))
    with pytest.raises(ValueError, match="duplicates"):
        replace(comparison, common_valid_fold_ids=("fold_001", "fold_001"))
    with pytest.raises(ValueError, match="SHA-256"):
        ReportIntegrity("bad")


def test_report_cross_section_reconciliation_rejects_mismatch() -> None:
    report = _report()
    with pytest.raises(ValueError, match="schema_version"):
        replace(report, schema_version="EvaluationReport.v0")
    with pytest.raises(ValueError, match="candidate IDs"):
        replace(report, configured_candidate_ids=tuple(reversed(_CANDIDATES)))
    with pytest.raises(ValueError, match="candidate reports"):
        replace(report, candidates=tuple(reversed(report.candidates)))
    with pytest.raises(ValueError, match="duplicate-free"):
        replace(report, planned_fold_ids=("fold_001", "fold_001"))
    with pytest.raises(ValueError, match="exact planned fold"):
        replace(report, planned_fold_ids=("fold_001", "fold_002"))

    first = report.candidates[0]
    changed_source = replace(first, source_build_id="other-build")
    with pytest.raises(ValueError, match="source build"):
        replace(report, candidates=(changed_source, *report.candidates[1:]))
    changed_plan = replace(first, evaluation_plan_hash="b" * 64)
    with pytest.raises(ValueError, match="evaluation plan"):
        replace(report, candidates=(changed_plan, *report.candidates[1:]))
    changed_feature = replace(first, feature_order=("move_level",))
    with pytest.raises(ValueError, match="feature order"):
        replace(report, candidates=(changed_feature, *report.candidates[1:]))
    changed_hash = replace(first, feature_selection_execution_hash="b" * 64)
    with pytest.raises(ValueError, match="feature-selection hashes"):
        replace(report, candidates=(changed_hash, *report.candidates[1:]))

    changed_selection = replace(report.feature_selection, final_features=("move_level",))
    with pytest.raises(ValueError, match="final features"):
        replace(report, feature_selection=changed_selection)
    changed_selection_hash = replace(
        report.feature_selection, feature_selection_execution_hash="b" * 64
    )
    with pytest.raises(ValueError, match="feature-selection hashes"):
        replace(report, feature_selection=changed_selection_hash)

    unknown_champion = replace(
        report.comparison,
        champion_candidate_id="unknown",
        ranked_candidate_ids=("unknown",),
    )
    with pytest.raises(ValueError, match="champion"):
        replace(report, comparison=unknown_champion)
    unknown_rank = replace(
        report.comparison,
        ranked_candidate_ids=(_CANDIDATES[0], "unknown"),
    )
    with pytest.raises(ValueError, match="ranking"):
        replace(report, comparison=unknown_rank)
    unknown_fold = replace(report.comparison, common_valid_fold_ids=("fold_999",))
    with pytest.raises(ValueError, match="common support"):
        replace(report, comparison=unknown_fold)


def test_integrity_hash_detects_missing_or_tampered_integrity() -> None:
    report = _report()
    with pytest.raises(ValueError, match="not sealed"):
        verify_report(report)
    with pytest.raises(ValueError, match="not sealed"):
        canonical_json_bytes(report)

    sealed = seal_report(report)
    digest = verify_report(sealed)
    assert len(digest) == 64
    corrupted = replace(sealed, integrity=ReportIntegrity("0" * 64))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_report(corrupted)
